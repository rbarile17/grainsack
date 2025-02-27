"""This module implements the LP-DIXIT protocol for evaluating explanations."""

import torch
from unsloth import FastLanguageModel

from grainsack.lp import rank
from grainsack.utils import load_kge_model

from transformers import pipeline

SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant.
Your response should be crisp, short and not repetitive.
Discard any preamble, explanation, greeting, or final consideration.
A triple is a statement <subject, predicate, object>.
The subject and the object are entities, and the predicate is a relation from the subject to the object.
Perform a Link Prediction task, given a query as an incomplete triple (subject, predicate, ?), predict the missing object that completes the triple making it a true statement.
Strict requirement: output solely the name of a single object entity, discard any explanation or other text. 
Correct format: Elizabeth_of_Bohemia
Incorrect format: The object entity is Elizabeth_of_Bohemia.
"""

USR_PROMPT = """
({subject}, {predicate}, ?)
{explanation}
"""

EXPLANATION_HOOK = """
In addition to the query, an explanation is provided.
An explanation is a set of triples relevant to the prediction.
Explanation:
"""


def select_examples(kg, predictions, ranks):
    """Select triples in the KG having the same predicate as the prediction and rank 1"""
    predictions = kg.id_triples(predictions)

    prediction_examples = []
    for prediction in predictions:
        triples_mask = kg.training_triples[:, 1] == prediction[1]
        ranks_mask = ranks == 1
        mask = triples_mask & ranks_mask
        examples = kg.training_triples[mask]

        prediction_examples.append(examples[torch.randperm(examples.size(0))[:10]])

    return prediction_examples


def format_prompt(explained_prediction, include_explanation=False):
    """Format the prompt for the LLM."""
    subject, predicate, _ = explained_prediction["prediction"]
    explanation = build_explanation(explained_prediction["explanation"]) if include_explanation else ""
    query = {"subject": subject, "predicate": predicate, "explanation": explanation}
    user_prompt = USR_PROMPT.format(**query)

    prompt = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]

    return prompt


def build_explanation(explanation):
    """Verbalize, i.e, transform from triple to text, the explanation for the LLM."""
    explanation = [f"({s}, {p}, {o})\n" for (s, p, o) in explanation]
    explanation = "\n".join(explanation)
    explanation = f"{EXPLANATION_HOOK}\n{explanation}"

    return explanation


# def parse_response(response):
#     """Parse the response from the LLM."""
#     response = response[0]["generated_text"][-1]["content"]
#     response = response.replace("\n", "")
#     response = response.replace(" ", "_")
#     response = response.replace(r"\_", "_")

#     return response


# def pipe(llm, tokenizer, prompts):
#     """Run the LLM pipeline."""
    
#     print()
#     prompts = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
#     inputs = inputs.to("cuda")
#     outputs = llm.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=64, use_cache=True)
#     outputs = [output[1:] for output in outputs] # Remove the duplicated begin token
#     outputs = [output[len(prompt):] for output, prompt in zip(outputs, prompts)]
#     outputs = tokenizer.batch_decode(outputs)

#     print(outputs)


def run_evaluate(explained_predictions, kg, kge_model_path, kge_config_path, eval_config):
    """Run the DIXIT task."""

    # llm_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    llm_id = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

    llm, tokenizer = FastLanguageModel.from_pretrained(model_name=llm_id, load_in_4bit=True)
    pipe = pipeline(task="text-generation", model=llm, tokenizer=tokenizer, truncation=True, padding=True)

    FastLanguageModel.for_inference(llm)
    # tokenizer.pad_token = tokenizer.eos_token  
    # tokenizer.padding_side = "left"

    predictions = [ex["prediction"] for ex in explained_predictions]
    gts = [o for _, _, o in predictions]

    print("Running pre-explanation simulations...")

    # kge_config = read_json(kge_config_path)
    # kge_model = load_kge_model(kge_model_path, kge_config, kg)
    # kge_model.eval()
    # kge_model.cuda()
    # ranks = rank(kge_model, kg.training_triples, filtr=[kg.training_triples]).cuda()
    # examples = select_examples(kg, predictions, ranks)

    # print(examples)

    
    prompts = [format_prompt(x) for x in explained_predictions]
    simulations = pipe(prompts, max_new_tokens=64, use_cache=True)
    simulations = [simulation[0]["generated_text"][-1]["content"] for simulation in simulations]

    print("Running post-explanation simulations...")
    prompts = [format_prompt(x, include_explanation=True) for x in explained_predictions]
    post_exp_simulations = pipe(prompts, max_new_tokens=64, use_cache=True)
    post_exp_simulations = [simulation[0]["generated_text"][-1]["content"] for simulation in post_exp_simulations]

    predictability_pre = [1 if o == gt else 0 for o, gt in zip(simulations, gts)]
    predictability_post = [1 if o == gt else 0 for o, gt in zip(post_exp_simulations, gts)]

    explanation_labels = [post - pre for post, pre in zip(predictability_post, predictability_pre)]

    for i in range(len(explained_predictions)):
        explained_predictions[i]["simulation"] = simulations[i]
        explained_predictions[i]["post_exp_simulation"] = post_exp_simulations[i]
        explained_predictions[i]["predictability_pre"] = predictability_pre[i]
        explained_predictions[i]["predictability_post"] = predictability_post[i]
        explained_predictions[i]["fs_label"] = explanation_labels[i]

    return explained_predictions
