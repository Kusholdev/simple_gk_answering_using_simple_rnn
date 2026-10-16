# Simple GK Question Answering Using Simple RNN

## 📌 Project Overview

This project implements a simple General Knowledge (GK) Question Answering System using a Recurrent Neural Network (RNN) with PyTorch.

The model takes a question as input, processes the text using tokenization and vocabulary indexing, and predicts an answer token based on the learned patterns in the training dataset.

The main goal of this project is to understand the fundamentals of Natural Language Processing (NLP), text preprocessing, word embeddings, and sequence modeling using a Simple RNN.

## 🎯 Objectives

- Understand basic NLP text preprocessing techniques.
- Build a vocabulary from questions and answers.
- Convert text into numerical representations.
- Create a custom PyTorch Dataset and DataLoader.
- Implement a Simple RNN model using PyTorch.
- Train the model using Cross-Entropy Loss and the Adam optimizer.
- Test the trained model with general knowledge questions.

## 🛠️ Technologies Used

- Python
- Pandas
- Regular Expressions (`re`)
- PyTorch
- Natural Language Processing (NLP)
- Simple Recurrent Neural Network (RNN)

## 📂 Dataset

The project uses a GK question-answering dataset stored in a CSV file named `gk_qna_dataset.csv`.

The dataset contains two main columns:

- `question`: The input question.
- `answer`: The corresponding answer.

Place the dataset in the appropriate directory and update the file path in the notebook if necessary.

## ⚙️ Project Workflow

### 1. Data Loading

The dataset is loaded into a Pandas DataFrame using `pd.read_csv()`.

### 2. Text Preprocessing

A custom tokenization function is used to preprocess the text by:

- Converting text to lowercase.
- Removing quotation marks.
- Removing punctuation.
- Removing extra whitespace.
- Splitting text into individual tokens.

### 3. Vocabulary Building

A vocabulary is created from the tokens in both questions and answers. Each unique token is assigned a numerical index.

The special token `<UNK>` represents unknown words that are not present in the vocabulary.

### 4. Text-to-Index Conversion

Questions and answers are converted into sequences of numerical indices using the vocabulary. These numerical sequences are used as inputs to the neural network.

### 5. Dataset and DataLoader

A custom `QADataset` class is implemented using PyTorch's `Dataset` class.

The `DataLoader` loads the question-answer pairs with a batch size of 1 and shuffles the data during training.

### 6. Simple RNN Architecture

The model consists of three main components:

- **Embedding Layer:** Converts token indices into dense vector representations.
- **RNN Layer:** Processes the embedded question sequence and produces a final hidden state.
- **Fully Connected Layer:** Maps the final hidden state to vocabulary-sized output scores.

#### Model Configuration

| Component | Configuration |
|---|---|
| Embedding dimension | 50 |
| Hidden size | 64 |
| RNN | Simple RNN |
| Batch size | 1 |
| Loss function | Cross-Entropy Loss |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Training epochs | 50 |

### 7. Model Training

The model is trained using the question sequences as input. During each training step, the predicted vocabulary scores are compared against the target token using Cross-Entropy Loss.

The optimizer updates the model parameters through backpropagation. Training loss is printed every 10 epochs.

**Note:** In the current implementation, the training target is only the first token of each answer. Therefore, the model learns to predict the first answer token rather than generating a complete answer.

### 8. Prediction

After training, a prediction function accepts a question as text, converts it into numerical indices, and passes it through the trained model.

The model selects the vocabulary token with the highest predicted score and converts that index back into a word.

Example test questions include:

- What is the capital of France?
- What is H2O commonly called?
- Which planet is known as the red planet?
- Which bird cannot fly?
- What is the capital of Japan?

The output is a single predicted token for each question.

## 🚀 How to Run the Project

### Prerequisites

Install Python and the required libraries:

```bash
pip install pandas torch
```

### Steps

1. Clone the repository:

   ```bash
   git clone <your-repository-url>
   ```

2. Navigate to the project directory:

   ```bash
   cd <your-project-folder>
   ```

3. Place `gk_qna_dataset.csv` in the expected location.

4. Open the Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `simple_gk_answering_using_simple_rnn.ipynb` and run the cells sequentially.

## 📚 Learning Outcomes

Through this project, I practiced:

- Basic NLP preprocessing and tokenization.
- Building and using a vocabulary.
- Converting textual data into tensors.
- Creating custom datasets with PyTorch.
- Understanding embedding layers and Simple RNN architecture.
- Training a neural network using backpropagation.
- Making predictions from a trained model.

## 🔮 Future Improvements

- Train the model to predict complete answers instead of a single token.
- Add padding to support batches of variable-length questions and answers.
- Improve text preprocessing and handle unknown words more effectively.
- Evaluate the model using suitable metrics.
- Experiment with LSTM, GRU, and attention-based architectures.
- Build a simple user interface for interactive question answering.

## 👨‍💻 Author

**Kushol Ranjan Sarker**

CSE Student | Machine Learning | Deep Learning | NLP
