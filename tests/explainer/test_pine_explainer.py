from transformers import AutoTokenizer, AutoModel
from torch.cuda import is_available as cuda_available

from pine.entity import Attribute, Entity, EntityPair
from pine.explainer.pine_explainer import (
    make_explanation,
)
from pine.matcher.magellan_matcher import make_magellan_matcher_func


def test_make_explanation():
    device = "cuda" if cuda_available() else "cpu"
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    dataset_name = "structured_amazon_google"
    model_root_dir = "examples/data/model"
    predict_proba_func = make_magellan_matcher_func(dataset_name, model_root_dir)

    entity_l = Entity(
        [
            Attribute("title", "iphone 12", "string"),
            Attribute("manufacturer", "apple", "string"),
            Attribute("price", 100.0, "Float64"),
        ]
    )
    entity_r = Entity(
        [
            Attribute("title", "iphone 13 iphone", "string"),
            Attribute("manufacturer", "apple", "string"),
            Attribute("price", 100.0, "Float64"),
        ]
    )
    entity_pair = EntityPair(entity_l, entity_r)
    topk = 5

    lime_result_pair, entity_pair_merged = make_explanation(
        entity_pair, predict_proba_func, topk, 
        model=bert_model, tokenizer=bert_tokenizer, n_sample=1000, random_state=0
        )

    assert len(lime_result_pair.attributions) <= topk
    assert len(lime_result_pair.attributions) == entity_pair_merged.segment_size()

    return
