from modal import App, Volume

app = App("plextract")

vol = Volume.from_name("plextract-vol", create_if_missing=True)