from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ANN_Project_Report.pdf"
VIZ = ROOT / "visualizations"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle", parent=styles["Title"],
    alignment=TA_CENTER, fontSize=19, leading=24, spaceAfter=14
))
styles.add(ParagraphStyle(
    name="Section", parent=styles["Heading1"],
    fontSize=14, leading=18, spaceBefore=8, spaceAfter=7
))
styles.add(ParagraphStyle(
    name="BodySmall", parent=styles["BodyText"],
    fontSize=9, leading=12, spaceAfter=6
))
styles.add(ParagraphStyle(
    name="SmallCenter", parent=styles["BodyText"],
    alignment=TA_CENTER, fontSize=8, leading=10
))

story = []

def para(text, style="BodySmall"):
    story.append(Paragraph(text, styles[style]))

def section(title):
    story.append(Paragraph(title, styles["Section"]))

def add_image(filename, width=15.5*cm, height=8.5*cm):
    path = VIZ / filename
    if path.exists():
        img = Image(str(path))
        img._restrictSize(width, height)
        story.append(img)
        story.append(Spacer(1, 6))

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2*cm, 1.1*cm, "ANN From Scratch Using NumPy")
    canvas.drawRightString(A4[0] - 2*cm, 1.1*cm, f"Page {doc.page}")
    canvas.restoreState()

story.append(Paragraph(
    "Artificial Neural Network<br/>From Scratch Using NumPy",
    styles["ReportTitle"]
))
para("<b>Student:</b> Satyam Kumar Singh")
para("<b>Course:</b> BCA")
para("<b>Dataset:</b> Iris flower classification")
para("<b>Project objective:</b> Implement and evaluate a feed-forward neural network "
     "using NumPy, including forward propagation, backpropagation, and gradient descent.")
story.append(Spacer(1, 8))

section("1. Dataset and preprocessing")
para("The Iris dataset contains 150 samples from three classes: Setosa, Versicolor, "
     "and Virginica. Each sample has four measurements: sepal length, sepal width, "
     "petal length, and petal width. The dataset was split using stratified sampling "
     "into 70% training, 15% validation, and 15% testing data. Features were "
     "standardized using the training-set mean and standard deviation only, helping "
     "prevent information from the validation and test sets from leaking into preprocessing. "
     "Class labels were converted to one-hot vectors for the output layer.")

section("2. Network architecture")
arch = [
    ["Layer", "Shape / units", "Purpose"],
    ["Input", "4 features", "Receives standardized flower measurements"],
    ["Hidden layer 1", "8 neurons, sigmoid", "Learns intermediate patterns"],
    ["Hidden layer 2", "6 neurons, sigmoid", "Learns additional combinations"],
    ["Output", "3 neurons, softmax", "Produces class probabilities"],
]
t = Table(arch, colWidths=[3.2*cm, 4.0*cm, 8.0*cm], repeatRows=1)
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dce6f1")),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 8),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("LEFTPADDING", (0,0), (-1,-1), 5),
    ("RIGHTPADDING", (0,0), (-1,-1), 5),
]))
story.append(t)
story.append(Spacer(1, 8))
para("The two hidden layers provide the model with the ability to learn nonlinear "
     "relationships between the four measurements. Sigmoid is used in the hidden "
     "layers and softmax converts the output scores into probabilities across the "
     "three flower classes. The model uses mini-batches of 16 and a learning rate "
     "of 0.01 in the baseline experiment.")

section("3. Forward propagation and backpropagation")
para("In forward propagation, each layer multiplies its input by a weight matrix "
     "and adds a bias. The hidden-layer outputs pass through the sigmoid function "
     "σ(z) = 1 / (1 + e⁻ᶻ). The final layer uses softmax so the three output "
     "probabilities sum to one. Cross-entropy loss measures the difference between "
     "the predicted probabilities and the one-hot target labels.")
para("During backpropagation, the output error is calculated from the predicted "
     "probabilities and true labels. The error is propagated backward through the "
     "weight matrices using the chain rule and the derivatives of the activation "
     "functions. Gradients for weights and biases are then used by gradient descent "
     "to update the parameters. Training history tracks loss and validation accuracy. "
     "Early stopping is configured to reduce unnecessary training when validation "
     "loss does not improve.")

section("4. Training results")
para("The baseline training run reached a peak validation accuracy of 95.45%. "
     "At epoch 1000, the recorded validation accuracy was 90.91% and validation "
     "loss was approximately 0.3018. The saved model evaluated on the held-out "
     "test split achieved 23 correct predictions out of 23, or 100% test accuracy.")
para("This is a result on one small test split, not proof that the model will achieve "
     "the same accuracy on other data. The small number of test samples means each "
     "prediction has a noticeable effect on the reported percentage.")
add_image("training_loss.png", height=5.7*cm)
add_image("validation_accuracy.png", height=5.7*cm)

section("5. Test evaluation")
para("The confusion matrix below shows the predictions reported by the saved model. "
     "Rows are actual classes and columns are predicted classes.")
cm_data = [
    ["Actual \\ Predicted", "Setosa", "Versicolor", "Virginica"],
    ["Setosa", "7", "0", "0"],
    ["Versicolor", "0", "8", "0"],
    ["Virginica", "0", "0", "8"],
]
t = Table(cm_data, colWidths=[4.3*cm, 3.1*cm, 3.1*cm, 3.1*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dce6f1")),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ALIGN", (1,0), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(t)
story.append(Spacer(1, 8))
add_image("confusion_matrix.png", height=7*cm)
section("6. Experiment comparison")
rows = [["Learning rate", "Activation", "Architecture", "Train loss",
         "Validation loss", "Validation accuracy"]]
csv_path = ROOT / "experiment_results.csv"
if csv_path.exists():
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append([
                    str(r.get("learning_rate", "")),
                    str(r.get("activation", "")),
                    str(r.get("architecture", "")),
                    f'{float(r.get("train_loss", "nan")):.3f}',
                    f'{float(r.get("validation_loss", "nan")):.3f}',
                    f'{float(r.get("validation_accuracy", r.get("val_accuracy", "nan")))*100:.2f}%'
                ])
            except (ValueError, TypeError):
                continue
else:
    para("experiment_results.csv was not found, so the experiment table could not be included.")

if len(rows) > 1:
    # Fit all rows on the page with a compact table.
    table = Table(rows, colWidths=[2.1*cm, 2.1*cm, 2.3*cm, 2.1*cm, 2.3*cm, 2.5*cm],
                  repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dce6f1")),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 6.5),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(table)

para("The experiments vary learning rate, hidden-layer activation, and architecture. "
     "A higher validation accuracy does not always mean a lower validation loss. "
     "Learning rate also affects convergence and generalization, so both metrics "
     "should be considered when comparing configurations.")

section("7. Limitations and conclusion")
para("This project demonstrates the core mechanics of a small neural network built "
     "with NumPy. Its main limitations are the small dataset and test split, and the "
     "fact that one run does not measure performance across multiple random splits. "
     "Further work could repeat experiments with different seeds, add regularization, "
     "and compare results across several train/test splits.")
para("The project helped connect the mathematical ideas of forward propagation, "
     "the chain rule, backpropagation, and gradient descent to a working classifier. "
     "The included scripts and CSV files make the training and experiment results "
     "available for review.")

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    rightMargin=1.7*cm, leftMargin=1.7*cm,
    topMargin=1.5*cm, bottomMargin=1.7*cm,
    title="ANN Project Report"
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Report created: {OUT}")
print(f"File size: {OUT.stat().st_size} bytes")
