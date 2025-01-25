from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

class RAGEvaluator:
    def __init__(self, dataset_path):
        self.dataset = self._load_dataset(dataset_path)
        self.vectorizer = TfidfVectorizer()

    def _load_dataset(self, dataset_path):
        with open(dataset_path, 'r') as file:
            data = json.load(file)
        return data["questions_and_answers"]

    def evaluate(self, local_model_responses, external_model_responses):
        if len(local_model_responses) != len(self.dataset) or len(external_model_responses) != len(self.dataset):
            raise ValueError("The number of responses must match the number of questions in the dataset.")

        local_scores = []
        external_scores = []

        for idx, entry in enumerate(self.dataset):
            expected_answer = entry["answer"]
            
            # Vectorizing expected and received answers
            local_response = local_model_responses[idx]
            external_response = external_model_responses[idx]

            answers = [expected_answer, local_response, external_response]
            tfidf_matrix = self.vectorizer.fit_transform(answers)

            # Cosine similarity
            expected_vector = tfidf_matrix[0]
            local_vector = tfidf_matrix[1]
            external_vector = tfidf_matrix[2]

            local_similarity = cosine_similarity(expected_vector, local_vector)[0][0]
            external_similarity = cosine_similarity(expected_vector, external_vector)[0][0]

            local_scores.append(local_similarity)
            external_scores.append(external_similarity)

        # Aggregated metrics
        local_avg_score = sum(local_scores) / len(local_scores)
        external_avg_score = sum(external_scores) / len(external_scores)

        return {
            "local_model": {
                "average_score": local_avg_score,
                "scores": local_scores
            },
            "external_model": {
                "average_score": external_avg_score,
                "scores": external_scores
            }
        }


if __name__ == '__main__':
    from got_rag
    # Example Usage:
    evaluator = RAGEvaluator("got_rag_test_dataset.json")
    local_responses = ["response1", "response2", ...]  # Replace with actual model responses
    external_responses = ["response1", "response2", ...]  # Replace with actual external model responses
    results = evaluator.evaluate(local_responses, external_responses)
    print(results)
