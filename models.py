import numpy as np
import re

class SimpleTfidfVectorizer:
    def __init__(self, max_features=3000):
        self.max_features = max_features
        self.vocabulary = {} # word -> index
        self.idf = None
        self.stop_words = set([
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
            "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
            "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
            "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", 
            "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", 
            "for", "with", "about", "against", "between", "into", "through", "during", "before", 
            "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
            "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", 
            "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
            "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", 
            "will", "just", "don", "should", "now"
        ])

    def clean_and_tokenize(self, text):
        text = str(text).lower()
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        # Keep only English letters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        words = text.split()
        
        # Simple stemming and stopword removal
        cleaned = []
        for w in words:
            if w not in self.stop_words:
                # Basic suffix stemming
                if len(w) > 2:
                    if w.endswith('ingly'): w = w[:-5]
                    elif w.endswith('ing'): w = w[:-3]
                    elif w.endswith('ly'): w = w[:-2]
                    elif w.endswith('ed'): w = w[:-2]
                    elif w.endswith('ies'): w = w[:-3] + 'y'
                    elif w.endswith('es') and not w.endswith('aes') and not w.endswith('ees') and not w.endswith('oes'): w = w[:-2]
                    elif w.endswith('s') and not w.endswith('us') and not w.endswith('ss') and not w.endswith('is') and not w.endswith('as'): w = w[:-1]
                cleaned.append(w)
        return cleaned

    def fit(self, corpus):
        word_counts = {}
        doc_freq = {}
        n_docs = len(corpus)

        for doc in corpus:
            tokens = self.clean_and_tokenize(doc)
            unique_tokens = set(tokens)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
            for token in unique_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Select top max_features by count
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = [w[0] for w in sorted_words[:self.max_features]]
        
        self.vocabulary = {word: idx for idx, word in enumerate(top_words)}
        
        # Compute IDF
        self.idf = np.zeros(len(self.vocabulary))
        for word, idx in self.vocabulary.items():
            df_val = doc_freq.get(word, 0)
            self.idf[idx] = np.log((1 + n_docs) / (1 + df_val)) + 1

    def transform(self, corpus):
        n_docs = len(corpus)
        n_features = len(self.vocabulary)
        X = np.zeros((n_docs, n_features), dtype=np.float32)
        
        for doc_idx, doc in enumerate(corpus):
            tokens = self.clean_and_tokenize(doc)
            tf = {}
            for token in tokens:
                if token in self.vocabulary:
                    tf[token] = tf.get(token, 0) + 1
            
            # Fill vector
            for word, count in tf.items():
                word_idx = self.vocabulary[word]
                X[doc_idx, word_idx] = count
            
            # Multiply by IDF
            X[doc_idx] = X[doc_idx] * self.idf
            
            # L2 normalization
            norm = np.linalg.norm(X[doc_idx])
            if norm > 0:
                X[doc_idx] = X[doc_idx] / norm
                
        return X

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)


class SimpleLogisticRegression:
    def __init__(self, lr=1.0, epochs=25, batch_size=512):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0

        for epoch in range(self.epochs):
            # Shuffle indices
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                xb = X[batch_indices]
                yb = y[batch_indices]

                # Forward pass
                linear = np.dot(xb, self.weights) + self.bias
                y_pred = self.sigmoid(linear)

                # Gradients
                dw = (1.0 / len(yb)) * np.dot(xb.T, (y_pred - yb))
                db = (1.0 / len(yb)) * np.sum(y_pred - yb)

                # Update weights
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
            
            # Evaluate performance on training batch
            linear = np.dot(X[:1000], self.weights) + self.bias
            y_pred = self.sigmoid(linear)
            train_acc = np.mean((y_pred >= 0.5) == y[:1000])
            print(f"Epoch {epoch+1}/{self.epochs} - Batch Accuracy: {train_acc*100:.2f}%")

    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        p = self.sigmoid(linear)
        # Return probability array of shape (M, 2)
        # Column 0: Negative probability (1 - p), Column 1: Positive probability (p)
        return np.column_stack((1 - p, p))

    def predict(self, X):
        # If Positive probability >= 0.5, predict 1, else 0
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
