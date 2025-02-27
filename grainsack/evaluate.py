"""This module implements the operation `evaluate`and the default prompt templates."""

import unsloth
import torch
from transformers import pipeline
from unsloth import FastLanguageModel

from grainsack.kg import KG
from grainsack.kge_lp import rank
from grainsack.utils import load_kge_model, read_json

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
{ranking}
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

def build_explanation(explanation):
    explanation = [f"({s}, {p}, {o})\n" for (s, p, o) in explanation]
    explanation = "\n".join(explanation)
    explanation = f"{EXPLANATION_HOOK}\n{explanation}"

    return explanation

def run_evaluate(explained_predictions, kg, kge_model_path, kge_config_path, eval_config):
    """Evaluate the given explanations based on the given data, models, and config.

    Evaluate the given explanations (each associated to a prediction) according to the given evaluation config
    and possibly (depending on the prompting method in the evaluation config) adopting the given KGE model and associated config.

    :param explained_predictions: The explanations (each associated to a prediction) to be evaluated.
    :type explained_predictions: dict
    :param kg: The KG used for the explanations and to be possibly (depending on the prompting method in the evaluation config) used for the evaluation.
    :type kg: KG 
    :param kge_model_path: The path to the KGE .pt model file used for the explanations and 
    to be possibly used (depending on the prompting method) for the evaluation.
    :type kge_model_path: pathlib.Path
    :param kge_config_path: The path to the KGE .json config file used for the explanations and
    to be possibly used (depending on the prompting method) for the evaluation.
    :type kge_config_path: pathlib.Path
    :param eval_config: The evaluation config dict containing:
    - the LLM id (from unsloth) and
    - the prompting method (zero_shot, zero_shot_constrained, few_shot, few_shot_constrained).
    :type eval_config: str
    :param output_path: The path to save the evaluations.
    :type output_path: pathlib.Path
    :return: The dictionary with the explained predictions and their evaluations in the fsv field
    :rtype: dict
    """

    def select_examples(kg: KG, predictions: torch.Tensor, ranks):
        predictions = kg.id_triples(predictions)

        prediction_examples = []
        for prediction in predictions:
            triples_mask = kg.training_triples[:, 1] == prediction[1]
            ranks_mask = ranks == 1
            mask = triples_mask & ranks_mask
            examples = kg.training_triples[mask]
            examples = examples[torch.randperm(examples.size(0))[:10]]
            examples = kg.label_triples(examples)

            prediction_examples.append(examples)

        return prediction_examples


    def format_prompt(explained_prediction, examples, ranking, include_explanation=False):
        subject, predicate, _ = explained_prediction["prediction"]
        explanation = build_explanation(explained_prediction["explanation"]) if include_explanation else ""

        examples_messages = []

        if examples is not None:
            example_subjects = [ex[0] for ex in examples]
            example_predicates = [ex[1] for ex in examples]
            example_objects = [ex[2] for ex in examples]
            queries = [
                {"subject": ex_subj, "predicate": ex_pred, "explanation": ""}
                for ex_subj, ex_pred in zip(example_subjects, example_predicates)
            ]
            user_prompts = [USER_PROMPT.format(**query) for query in queries]

            for i in range(len(examples)):
                examples_messages.append({"role": "user", "content": user_prompts[i]})
                examples_messages.append({"role": "assistant", "content": example_objects[i]})

        query = {"subject": subject, "predicate": predicate, "explanation": explanation}
        user_prompt = USER_PROMPT.format(**query)

        system_prompt = SYSTEM_PROMPT.format(
            ranking=f"The entity name that you provide must be in the following list of entity names: {ranking}" if ranking is not None else ""
        )

        prompt = [{"role": "system", "content": system_prompt}] + examples_messages + [{"role": "user", "content": user_prompt}]

        return prompt

    llm_id = eval_config.get("llm_id", "unsloth/Llama-3.2-1B-Instruct-bnb-4bit")
    prompting = eval_config.get("prompting", "zero-shot")

    llm, tokenizer = FastLanguageModel.from_pretrained(model_name=llm_id, load_in_4bit=True)
    pipe = pipeline(task="text-generation", model=llm, tokenizer=tokenizer, truncation=True, padding=True)
    FastLanguageModel.for_inference(llm)

    predictions = [ex["prediction"] for ex in explained_predictions]
    gts = [o for _, _, o in predictions]

    rankings = None
    examples = None

    if prompting in ["zero_shot_constrained", "few_shot", "few_shot_constrained"]:
        kge_config = read_json(kge_config_path)
        kge_model = load_kge_model(kge_model_path, kge_config, kg)
        kge_model.eval()
        kge_model.cuda()
    if prompting in ["zero_shot_constrained", "few_shot_constrained"]:
        predictions_tensor = torch.tensor(kg.id_triples(predictions))
        object_scores = kge_model.predict_t(predictions_tensor)
        rankings = torch.argsort(object_scores, dim=1, descending=True)
        rankings = [[kg.id_to_entity[ranking.item()] for ranking in rankings[i]] for i in range(len(predictions))]
    if prompting in ["few_shot", "few_shot_constrained"]:
        ranks = rank(kge_model, kg.training_triples, filtr=[kg.training_triples]).cuda()
        examples = select_examples(kg, predictions, ranks)

    print("Running pre-explanation simulations...")
    prompts = [
        format_prompt(
            explained_predictions[i],
            None if examples is None else examples[i],
            None if rankings is None else rankings[i][:100],
        )
        for i in range(len(predictions))
    ]
    simulations = pipe(prompts, max_new_tokens=64, use_cache=True)
    simulations = [simulation[0]["generated_text"][-1]["content"] for simulation in simulations]

    print("Running post-explanation simulations...")
    prompts = [
        format_prompt(
            explained_predictions[i],
            None if examples is None else examples[i],
            None if rankings is None else rankings[i][:100],
            include_explanation=True,
        )
        for i in range(len(predictions))
    ]
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
        explained_predictions[i]["fsv"] = explanation_labels[i]

    return explained_predictions
