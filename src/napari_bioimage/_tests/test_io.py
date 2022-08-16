from napari_bioimage._io import extract_urls, load_model_rdf


def test_extract_urls(data_dir):
    model = load_model_rdf(data_dir / "rdf2.yaml")
    urls = extract_urls(model)
    assert len(urls) == 2
