import pickle as pk

class DecisionTreeModel:

    def __init__(self):
        self.model_pipe = pk.load(open("./decision_tree_pipe_spotify.pkl", "rb"))
        self.scaler = {
            "z_score": pk.load(open("./encoder_z_score.pkl", "rb")),
            "min_max": pk.load(open("./encoder_min_max.pkl", "rb")),
            "one_hot": pk.load(open("./encoder_one_hot.pkl", "rb")),
            "ta_encoder": pk.load(open("./encoder_ta.pkl", "rb"))
        }
        self.model_manual = pk.load(open("./decision_tree.pkl", "rb"))

    def predict(self, df):
        return self.model_pipe.predict(df)
