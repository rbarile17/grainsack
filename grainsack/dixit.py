import time

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from grainsack import FS_DETAILS_PATH, FS_METRICS_PATH, EXPLANATIONS_PATH

from .utils import read_json, write_json

system_prompt = """
You are a helpful, respectful and honest assistant.
Your response should be crisp, short and not repetitive.
Discard any preamble, explanation, greeting, or final consideration.
A triple is a statement <subject, predicate, object>.
The subject and the object are entities, and the predicate is a relation between the subject and the object.
Perform a Link Prediction task, given a query as an incomplete triple (subject, predicate, ?), predict the missing object that completes the triple and makes it a true statement.
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


def format_prompt(explained_prediction, include_explanation=False):
    subject, predicate, _ = explained_prediction["prediction"]
    explanation = build_explanation(explained_prediction["explanation"]) if include_explanation else ""
    query = {"subject": subject, "predicate": predicate, "explanation": explanation}
    user_prompt = user_prompt_template.format(**query)

    prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    return prompt


def build_explanation(explanation):
    explanation = [f"({s}, {p}, {o})\n" for (s, p, o) in explanation]
    explanation = "\n".join(explanation)
    explanation = f"{explanation_hook}\n{explanation}"

    return explanation


def parse_response(response):
    response = response[0]["generated_text"][-1]["content"]
    response = response.replace("\n", "")
    response = response.replace(" ", "_")
    response = response.replace("\_", "_")

    return response


def run_dixit_task(kg, model, method, mode, parameters, llm):
    exp_name = f"{method}_{mode}_{model}_{kg}_{llm}_{parameters}"

    llm_id = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(llm_id, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    llm = AutoModelForCausalLM.from_pretrained(llm_id, device_map="auto")

    pipe = pipeline("text-generation", model=llm, tokenizer=tokenizer, batch_size=32)

    start = time.time()

    explained_predictions = read_json(EXPLANATIONS_PATH / f"{method}_{mode}_{model}_{kg}.json")

    predictions = [explained_prediction["prediction"] for explained_prediction in explained_predictions]
    gts = [o for _, _, o in predictions]

    print("Running pre-explanation simulations...")
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

    write_json(explained_predictions, FS_DETAILS_PATH / f"{exp_name}.json")
