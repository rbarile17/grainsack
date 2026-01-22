"""This module implements the operation `evaluate` and the default prompt templates."""

import torch
from transformers import pipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from grainsack.utils import read_json

from grainsack import KGS_PATH

SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant.
Your response should be crisp, short and not repetitive.
Discard any preamble, explanation, greeting, or final consideration.
A triple is a statement <subject, predicate, object>.
The subject and the object are entities, and the predicate is a relation from the subject to the object.
Perform a Link Prediction task, given a query as an incomplete triple (subject, predicate, ?), predict the missing object that completes the triple making it a true statement.
Strict requirement: output solely the name of a single object entity, discard any explanation or other text.
Correct format: Elizabeth_of_Bohemia
Incorrect format: The object entity is Elizabeth_of_Bohemia
"""

USER_PROMPT = """
({subject}, {predicate}, ?)
{explanation}
"""

EXPLANATION_HOOK = """
In addition to the query, an explanation is provided.
An explanation is a set of triples relevant to the prediction.
Explanation:
"""


def run_evaluate(explanations, kg_name):
    """Evaluate the given explanations based on the given data, models, and config.

    Evaluate the given explanations (each associated to a prediction) according to the given evaluation config
    and possibly (depending on the prompting method in the evaluation config) adopting the given KGE model and associated config.

    :param explanations: The explanations (each associated to a prediction) to be evaluated.
    :type explanations: dict
    """

    def format_prompt(explained_prediction, include_explanation=False):
        subject, predicate, _ = explained_prediction["prediction"]
        subject = ind_labels.get(subject, subject.split("/")[-1])
        predicate = prop_labels[predicate]

        explanation = ""
        if include_explanation:
            explanation = explained_prediction["explanation"]
            explanation = [
                (ind_labels.get(s, s.split("/")[-1]), prop_labels[p], ind_labels.get(o, o.split("/")[-1]))
                for (s, p, o) in explanation
            ]
            explanation = [f'("{s}", {p}, "{o}")\n' for (s, p, o) in explanation]
            explanation = "\n".join(explanation)
            explanation = f"{EXPLANATION_HOOK}\n{explanation}"

        query = {"subject": subject, "predicate": predicate, "explanation": explanation}
        user_prompt = USER_PROMPT.format(**query)

        prompt = [{"role": "system", "content": SYSTEM_PROMPT}] + [{"role": "user", "content": user_prompt}]

        return prompt

    ind_labels = read_json(KGS_PATH / kg_name / "ind_labels.json")
    prop_labels = read_json(KGS_PATH / kg_name / "prop_labels.json")

    llm_id = "Meta-Llama-3.1-70B-Instruct"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    llm = AutoModelForCausalLM.from_pretrained(llm_id, quantization_config=quantization_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(llm_id, padding_side="left")

    pipe = pipeline(
        task="text-generation", model=llm, tokenizer=tokenizer, truncation=True, padding=True, pad_token_id=tokenizer.eos_token_id
    )

    pipe.tokenizer.pad_token_id = llm.config.eos_token_id[0]

    predictions = [ex["prediction"] for ex in explanations]
    gts = [o for _, _, o in predictions]
    gts = [ind_labels.get(gt, gt.split("/")[-1]) for gt in gts]

    print("Running pre-explanation simulations...")
    prompts = [format_prompt(explanations[i]) for i in range(len(predictions))]
    simulations = pipe(prompts, max_new_tokens=64, use_cache=True, batch_size=8)
    simulations = [simulation[0]["generated_text"][-1]["content"] for simulation in simulations]

    print("Running post-explanation simulations...")
    prompts = [format_prompt(explanations[i], include_explanation=True) for i in range(len(predictions))]
    post_exp_simulations = pipe(prompts, max_new_tokens=64, use_cache=True, batch_size=8)
    post_exp_simulations = [simulation[0]["generated_text"][-1]["content"] for simulation in post_exp_simulations]

    predictability_pre = [1 if o == gt else 0 for o, gt in zip(simulations, gts)]
    predictability_post = [1 if o == gt else 0 for o, gt in zip(post_exp_simulations, gts)]

    explanation_labels = [post - pre for post, pre in zip(predictability_post, predictability_pre)]

    for i in range(len(explanations)):
        explanations[i]["simulation"] = simulations[i]
        explanations[i]["post_exp_simulation"] = post_exp_simulations[i]
        explanations[i]["predictability_pre"] = predictability_pre[i]
        explanations[i]["predictability_post"] = predictability_post[i]
        explanations[i]["fsv"] = explanation_labels[i]

    return explanations
