from sentence_transformers import SentenceTransformer, util
import Levenshtein


class KeywordMatcher:
    def __init__(self, model_name="MiniLM"):
        """初始化嵌入模型"""
        self.model_name = model_name
        self.model = self._initialize_model(model_name)
        self.input_embedding = None

    def _initialize_model(self, model_name):
        """根據模型名稱初始化嵌入模型"""
        if model_name == "MiniLM":
            print("使用 Sentence Transformers 作為嵌入模型")
            return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        elif model_name == "Levenshtein":
            print("使用 Levenshtein")
            return Levenshtein.ratio
        else:
            raise ValueError(f"不支持的嵌入模型: {model_name}")

    def encode_keywords(self, keywords):
        """向量化關鍵字列表"""
        if self.model_name == "MiniLM":
            return self.model.encode(keywords)
        elif self.model_name == "Levenshtein":
            return keywords

    def compute_similarity(self, input_text, keyword_embeddings):
        """計算輸入文本與關鍵字的相似度"""
        if self.model_name == "MiniLM":
            if self.input_embedding is None:
                self.input_embedding = self.model.encode(input_text)
            return util.cos_sim(self.input_embedding, keyword_embeddings)
        elif self.model_name == "Levenshtein":
            return [[self.model(input_text, k) for k in keyword_embeddings]]


if __name__ == "__main__":
    matcher = KeywordMatcher(model_name="MiniLM")

    keywords = ["語音轉文字", "錄音", "現場人數", "請開始錄音"]
    keyword_embeddings = matcher.encode_keywords(keywords)

    input_text = "請開始錄音"
    similarities = matcher.compute_similarity(input_text, keyword_embeddings)

    for keyword, similarity in zip(keywords, similarities[0]):
        print(f"- {keyword}: {similarity:.4f}")
