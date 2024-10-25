from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

def plot_model_accuracy(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
def plot_model_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    


def evaluate_model_performance(predictions, test_Y):
    # Get the predictions from the model
    predictions = [1 if p > 0.5 else 0 for p in predictions]


    # Accuracy, Precision, Recall, F1 Score
    acc = accuracy_score(test_Y, predictions)
    precision = precision_score(test_Y, predictions)
    recall = recall_score(test_Y, predictions)
    f1 = f1_score(test_Y, predictions)
    bacc = balanced_accuracy_score(test_Y, predictions)

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Balanced Accuracy: {bacc}")

    # Confusion Matrix
    cm = confusion_matrix(test_Y, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(test_Y, predictions, target_names=['Negative', 'Positive']))


