import sys
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

class PredictionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Heart Failure Research System')
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.result_label = QLabel('Prediction result will be shown here.')
        layout.addWidget(self.result_label)

        predict_button = QPushButton('Predict', self)
        predict_button.clicked.connect(self.predict)
        layout.addWidget(predict_button)

        self.setLayout(layout)

    def predict(self):
        data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

        X = data.iloc[:, 0:-1].values
        y = data.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)
        model = DecisionTreeClassifier(max_depth=2, random_state=42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)


        count = 0
        for data in range(len(y_pred)):
            if y_pred[data] != y_test[data]:
                count = count + 1


        error_percentage = count/len(y_test) * 100
        print(f"{error_percentage}%")

def main():
    app = QApplication(sys.argv)
    ex = PredictionApp()
    ex.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()



