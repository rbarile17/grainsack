# from pytest import fixture

# from conftest import kg

# import torch

# from grainsack.sift import topology_sift, criage_sift


# @fixture
# def prediction(kg):
#     prediction = ("mali", "locatedin", "africa")
#     prediction = kg.id_triple(prediction)
#     prediction = torch.tensor(prediction).cuda()

#     return prediction


# def test_topology_sift(kg, prediction):

# qui in realtà devo testare la fitness e poi testare questo con una fitness dummy

#     selected_triples = topology_sift(kg, prediction, k=2)

#     correct_triples = [
#         ("algeria",	"neighbor", "mali"),
#         ("burkina_faso", "neighbor", "mali"),
#     ]

#     correct_triples = torch.tensor(kg.id_triples(correct_triples)).cuda()

#     assert torch.equal(selected_triples, correct_triples)


# def test_criage_filter(kg, prediction):
#     selected_triples = criage_sift(kg, prediction, k=2)

#     correct_triples = [("algeria", "locatedin", "africa"), ("algeria", "neighbor", "mali")]
#     correct_triples = torch.tensor(kg.id_triples(correct_triples)).cuda()

#     assert torch.equal(selected_triples, correct_triples)
