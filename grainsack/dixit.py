"""This module implements the DIXIT task for evlaution of explanations."""

import random

import torch

from grainsack import DIXIT_PATH, EXPLANATIONS_PATH, LP_CONFIGS_PATH
from grainsack.kg import KG
from grainsack.lp import evaluate
from grainsack.utils import load_model
from unsloth import FastLanguageModel


from .utils import read_json, write_json

system_prompt = """
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

user_prompt_template = """
({subject}, {predicate}, ?)
{explanation}
"""

explanation_hook = """
In addition to the query, an explanation is provided.
An explanation is a set of triples relevant to the prediction.
Explanation:
"""


def select_examples(kg, predictions, ranks):
    """select triples in the KG having the same predicate as the prediction and rank 1"""
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
    user_prompt = user_prompt_template.format(**query)

    prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    return prompt


def build_explanation(explanation):
    """Verablize, i.e, transform from triple to text, the explanation for the LLM."""
    explanation = [f"({s}, {p}, {o})\n" for (s, p, o) in explanation]
    explanation = "\n".join(explanation)
    explanation = f"{explanation_hook}\n{explanation}"

    return explanation


def parse_response(response):
    """Parse the response from the LLM."""
    response = response[0]["generated_text"][-1]["content"]
    response = response.replace("\n", "")
    response = response.replace(" ", "_")
    response = response.replace("\_", "_")

    return response


def run_dixit_task(model, kg, method, explanation_kwargs, llm):
    """Run the DIXIT task."""
    kwargs_str = "_".join(str(v) if v is not None else "null" for v in explanation_kwargs.values())
    exp_name = DIXIT_PATH / f"{model}_{kg}_{method}_{kwargs_str}_{llm}.json"

    # llm_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    llm_id = "unsloth/tinyllama"

    llm, tokenizer = FastLanguageModel.from_pretrained(model_name=llm_id, load_in_4bit=True)

    FastLanguageModel.for_inference(llm)
    # inputs = tokenizer([prompt.format("Continue the fibonnaci sequence.", "1, 1, 2, 3, 5, 8", "")], return_tensors="pt")
    # inputs = inputs.to("cuda")

    # outputs = llm.generate(**inputs, max_new_tokens=64, use_cache=True)
    # print(tokenizer.batch_decode(outputs))

    explained_predictions = read_json(EXPLANATIONS_PATH / f"{model}_{kg}_{method}_{kwargs_str}.json")

    predictions = [explained_prediction["prediction"] for explained_prediction in explained_predictions]
    gts = [o for _, _, o in predictions]

    print("Running pre-explanation simulations...")

    # lp_config_path = LP_CONFIGS_PATH / f"{model}_{kg}.json"
    # lp_config = read_json(lp_config_path)
    # kg = KG(kg=kg)
    # model = load_model(lp_config, kg)
    # model.eval()
    # model.cuda()
    # ranks = evaluate(model, kg.training_triples, filter=[kg.training_triples]).cuda()
    # examples = select_examples(kg, predictions, ranks)

    prompts = [format_prompt(x) for x in explained_predictions]
    simulations = [parse_response(response) for response in pipe(prompts)]

    print("Running post-explanation simulations...")
    prompts = [format_prompt(x, include_explanation=True) for x in explained_predictions]
    post_exp_simulations = [parse_response(response) for response in pipe(prompts)]

    predictability_pre = [1 if o == gt else 0 for o, gt in zip(simulations, gts)]
    predictability_post = [1 if o == gt else 0 for o, gt in zip(post_exp_simulations, gts)]

    explanation_labels = [post - pre for post, pre in zip(predictability_post, predictability_pre)]

    for i in range(len(explained_predictions)):
        explained_predictions[i]["simulation"] = simulations[i]
        explained_predictions[i]["post_exp_simulation"] = post_exp_simulations[i]
        explained_predictions[i]["predictability_pre"] = predictability_pre[i]
        explained_predictions[i]["predictability_post"] = predictability_post[i]
        explained_predictions[i]["fs_label"] = explanation_labels[i]

    write_json(explained_predictions, exp_name)
