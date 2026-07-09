# CIS 5526 Machine Learning — Merged Lecture Notes on Supervised Learning and k-Nearest Neighbors

**Sources merged:** instructor handwritten PDF notes, plus the two uploaded VTT lecture transcripts from 2025-09-03 and 2025-09-10.  
**Scope:** this document follows the KNN-related lecture sequence: the supervised-learning setup, the mental picture of classification/regression, the discovery and formalization of k-nearest neighbors, distance choices, data preprocessing, train/test evaluation, accuracy measures, live KNN demos, and the curse of dimensionality. The September 10 transcript begins after the instructor had already started the distance-metric discussion, so the missing beginning of that class is preserved from the written notes.

---

## 1. Transition into Supervised Learning

### Written slide / notes content

The instructor begins the handwritten notes with the title **Supervised Learning**.

The formal setup is:

\[
D = \{(x_i, y_i),\ i = 1,\dots,N\}
\]

where:

- \(N\) is the number of examples.
- \(x_i\) is the feature vector for the \(i\)-th example:

\[
x_i = (x_{i1}, x_{i2}, \dots, x_{iM})
\]

- \(M\) is the number of features.
- \(x_{ij}\) means: the \(j\)-th feature of the \(i\)-th example.
- \(y_i\) is the label of the \(i\)-th example.

The slide draws a conceptual pipeline:

1. There is a **World**.
2. We obtain a dataset \(D\) by **sampling** from the world.
3. From \(D\), we perform **learning** to create a model \(f\).
4. The model maps an input \(x\) to a predicted output:

\[
y = f(x)
\]

5. We then **apply** the learned model back in the world.

The supervised-learning task is written as:

> Given \(D\), learn \(f\) that accurately predicts \(y\) from \(x\).

### Spoken explanation

The instructor introduces this as the beginning of the core supervised-learning part of the course. He explains that, for much of the course, whenever he says “dataset,” he means this kind of labeled dataset: a collection of examples, each with features and a label.

He emphasizes that the dataset does not appear magically. It is sampled from some larger real-world population or process. Because of that, whenever we receive a dataset, we should ask where it came from and how it was sampled. Sampling can strongly affect whether later conclusions are meaningful.

He gives an example using cars: a dataset might contain car models sold in the United States between certain years, with features such as horsepower, weight, displacement, and so on. The label might be miles per gallon. In that case, the task is to learn a function that predicts fuel economy from the other car attributes.

He also gives a survey analogy. If one wants to estimate opinions in the United States, one cannot ask everyone, so one samples people. But sampling people randomly is difficult. Calling random phone numbers introduces bias because people without phones are excluded, and people who refuse to answer may differ systematically from those who participate. This is why the data-generation and sampling process matters.

The instructor says that, in this course, we will often assume the dataset is a random or representative sample from some underlying world or distribution, even though in real projects this assumption needs to be questioned.

### Verbal examples and asides not on the slide

- The instructor tells students that he prefers writing on an iPad instead of showing slides because he wants them to take notes. He says the act of taking notes helps learning.
- A student asks whether the handwritten notes will be shared. The instructor says he will share materials periodically, but not immediately after every lecture, because he wants students to listen and take their own notes.
- The instructor mentions spam filtering as an example of supervised learning: an email is converted into features \(x\), and the model predicts whether it is spam.
- He mentions that a model can be deployed “back in the world,” meaning it is not just something evaluated inside the classroom or dataset; it is supposed to be useful on new real cases.

---

## 2. Three Core Questions in Supervised Learning

### Written slide / notes content

The notes list three major questions:

### 1. What is \(f\)?

The model \(f\) may be a **mathematical function**. Example:

\[
f(x) = 3 + 2x_1 - 7x_2
\]

This is a linear function of the features and leads to **linear regression**.

The notes also state that \(f(x)\) could be a **nonlinear function** of features, which leads to **neural networks / deep learning**.

Alternatively, \(f\) may be a **computer algorithm**, including:

- search-based algorithms, such as **k-nearest neighbors**,
- rule-based algorithms, such as **decision trees**.

### 2. How good is \(f\)?

To compare models, we need an **accuracy measure** or a **loss function**. The notes explicitly ask:

- Is \(f_1\) better than \(f_2\)?
- We need to define how model quality is measured.

### 3. How do we learn \(f\) from \(D\)?

The final core question is the learning question: once we know what kind of model we want and how we will judge it, how do we actually create it from the dataset?

The notes also list the major types of supervised learning:

- **Classification:** \(y\) is categorical.
- **Regression:** \(y\) is numerical, \(y \in \mathbb{R}\).

### Spoken explanation

The instructor frames these as the three big questions that organize the course.

First, we must decide what type of object the model is. It might be a mathematical formula. A linear formula such as \(3 + 2x_1 - 7x_2\) is a simple example and belongs to linear regression. If we allow nonlinear mathematical functions, we move toward neural networks and deep learning.

The instructor then broadens the idea of a model. A model does not have to be a clean algebraic function. It can be a computer algorithm that takes an input and returns a prediction. This matters because a computer program can handle things that are awkward inside a mathematical formula, such as strings, categories, or rule-based logic.

He identifies KNN as an example of a search-based algorithm: the algorithm searches the dataset for nearby examples. Decision trees are introduced as rule-based algorithms because they make if-else decisions based on feature values.

Second, he explains that a model cannot be called “good” without a measurement rule. If two people build two different spam filters, we need a formula or procedure to decide which one is better. That is the role of accuracy measures or loss functions.

Third, after deciding what kind of model we want and how we will evaluate it, we can ask the central machine-learning question: how do we learn or create \(f\) from the data \(D\)?

### Verbal examples and asides not on the slide

- The instructor uses a “customer” example: in a real project, the customer may define what should be predicted. For example, if the customer wants MPG predicted, the data scientist may not always ask why; they build the requested predictor.
- He mentions a hypothetical competition where a company offers a large prize for the best spam filter. That motivates why we need a precise comparison measure between models.
- He points out that machine-learning terminology can sound intimidating, but terms like “loss function” simply mean a penalty for being wrong.

---

## 3. Classification and Regression: Mental Pictures

### Written slide / notes content

The next page is titled **Mental Picture of S.L.**

### Classification

The notes show a two-dimensional feature space with axes \(x_1\) and \(x_2\). The examples are marked with class labels, such as \(+\) and \(-\), for binary classification:

\[
y \in \{+,-\}
\]

The drawing shows different possible decision boundaries, labeled as possible models \(f_1(x)\) and \(f_2(x)\). One boundary separates the positive and negative examples better than the other.

### Regression

The notes show a scatterplot with \(x_1\) on the horizontal axis and \(y\) on the vertical axis. Two possible functions, \(f_1(x)\) and \(f_2(x)\), are drawn through the data. One is a straight line; another is a nonlinear curve.

### Spoken explanation

The instructor says these mental pictures are useful because they let us reason visually about what a model is doing.

For classification, he restricts the picture to two features because humans can visualize two dimensions easily. Each point is an example, and instead of drawing dots only, he marks the points by their labels: plus or minus. If there were three classes, one could imagine using three different symbols, such as circles, triangles, and stars.

The model can then be imagined as a boundary in feature space. One boundary may say that all points above it are plus and all points below it are minus. Another boundary might be curved or placed differently. We can then visually debate which boundary seems to classify the examples better.

For regression, the instructor explains that the same symbol-based picture does not work as well because \(y\) can take infinitely many numerical values. Instead, he uses the vertical axis to show the label value \(y\), and the horizontal axis to show one feature \(x_1\). A regression model becomes a curve or line trying to pass through the data cloud.

The key idea is that visualizing candidate functions helps us ask the same core questions again: which function is better, how do we measure better, and how do we learn the better function automatically?

### Verbal examples and asides not on the slide

- The instructor says he will reuse this visual picture many times when explaining algorithms.
- He notes that the two-dimensional picture is a simplification, because real datasets may have many features, but the picture is still useful for intuition.
- A student asks whether mathematical functions are assumed to be continuous, differentiable, or smooth. The instructor says that such assumptions are often made, but not always; machine learning methods differ in what they assume about the data and the model.

---

## 4. Discovering the First Algorithm: Nearest-Neighbor Reasoning

### Written slide / notes content

The notes show a binary-classification scatterplot with plus and minus examples. A new unlabeled point is drawn near several negative examples.

The handwritten annotation says:

> We think it should be “-” because all its near neighbors are “-”.

This leads to:

> k-nearest neighbors / KNN

The underlying principle is written as:

> “guilt by association”

### Spoken explanation

The instructor asks the class to imagine a new example appearing in the feature space. It has no label yet. He asks students what label they would guess.

Most students think the new point should be negative, because it is surrounded by negative examples. One student suggests it could be positive because of an outlier-like pattern, but the instructor uses that answer to emphasize that the reasoning itself is important: everyone is looking at nearby examples.

The class’s reasoning becomes the first machine-learning algorithm. If a new example is near examples of a particular class, we predict that it belongs to that class. This is the basis of k-nearest neighbors.

The instructor calls the principle “guilt by association.” If you live among negative examples, the algorithm treats you as negative; if you live among positive examples, it treats you as positive. He notes that this is a kind of bias or prior assumption: nearby things tend to be similar. He says that, in some form, most machine-learning algorithms rely on this kind of association principle.

### Verbal examples and asides not on the slide

- The instructor compares the algorithmic bias to social bias: if people judge you by your neighborhood before knowing you personally, that is guilt by association. KNN does something similar mathematically.
- He says the true label of the new example may be revealed later or may never be known, but the algorithm still has to make a prediction.
- At the end of the first class segment, he says that the next lecture will formalize KNN, write pseudocode, discuss computational cost, and introduce core machine-learning terminology through this simple algorithm.

---

## 5. Formal Setup of k-Nearest Neighbors

### Written slide / notes content

The notes define the KNN setup:

- Given a labeled dataset \(D\).
- Given a new unlabeled example \(x\).
- Objective: predict the class/label \(y\) of \(x\).

The KNN idea is written as three steps:

1. Find all distances between \(x\) and the examples in \(D\).
2. Identify the \(k\) nearest neighbors.
3. Use the labels of the \(k\) neighbors to predict \(y\).

For binary classification, the common approach is:

> majority voting

The notes also label KNN as the first supervised-learning algorithm in the course.

### Spoken explanation

The instructor explains that KNN is conceptually very simple. It does not first fit a formula like linear regression. Instead, it keeps the training data and, when a new point arrives, searches through the existing examples.

The first operation is to measure how far the new point is from every example in the training set. Then the algorithm selects the closest \(k\) examples. Finally, it looks at their labels. In classification, the predicted label is usually the majority label among those neighbors.

The instructor repeatedly emphasizes that the simplicity of KNN is deceptive. Even this simple algorithm immediately raises practical decisions: how many neighbors should be used, and what does “near” mean?

### Verbal examples and asides not on the slide

- The instructor says KNN is almost the simplest algorithm, but even here there are many things to worry about.
- He introduces the word **hyperparameter** for choices made before the model is created, such as \(k\) and the distance function.
- He says that \(k\) should not be too small and not too large, and that choosing it involves trial and error.

---

## 6. KNN Pseudocode and Computational Cost

### Written slide / notes content

The notes give pseudocode for KNN:

```text
# find all distances from x
for i = 1:N
    d(i) = dist(x, x_i)

# example distance: Euclidean distance
dist(x, x_i) = sqrt((x_1 - x_{i1})^2 + (x_2 - x_{i2})^2 + ... + (x_M - x_{iM})^2)

# identify k nearest neighbors
index = sort(d)

# majority voting
pred = majority(y[index(1:k)])
```

The notes mark the computational costs:

- Computing one Euclidean distance costs \(O(M)\), because it uses all \(M\) features.
- Computing distances to all \(N\) training examples costs \(O(MN)\).
- Sorting distances costs \(O(N\log N)\).
- Majority vote over \(k\) labels costs \(O(k)\).

The total is written approximately as:

\[
O(MN) + O(N\log N) + O(k) \approx O(MN)
\]

The notes emphasize that KNN must scan the whole dataset \(D\), which is why KNN is considered computationally costly in machine learning. A desirable cost would be closer to \(O(M)\), independent of the number of training examples.

### Spoken explanation

The instructor explains that the expensive part is not the majority vote; the expensive part is searching through the training set. For every new point, KNN must compare it against all stored examples. If the dataset is large, this can be slow.

He highlights a practical tension: KNN wants lots of data because more data gives better neighborhoods, but lots of data also makes prediction more expensive because every query requires searching through the data.

The instructor also notes that, in the computational-cost expressions, \(M\) is the number of features and \(N\) is the number of examples. This distinction will appear throughout the course.

### Verbal examples and asides not on the slide

- The transcript does not preserve the full spoken explanation at the very start of the September 10 class because the recording began late. The written notes preserve the pseudocode and cost analysis.
- In later discussion, a student asks whether alternative distances reduce complexity. The instructor says that Euclidean, Manhattan, and general \(L_p\)-type distances all still require looking across the features, so they have the same basic \(O(M)\) per-distance cost.

---

## 7. Practical Hyperparameters: \(k\) and Distance

### Written slide / notes content

The notes list two practical questions:

1. What is \(k\)?
2. What is \(\operatorname{dist}(\cdot)\)?

These are labeled:

> Hyperparameters — decisions that should be made before creating \(f\)

The notes add that \(k\) should be:

- not too small,
- not too large,
- chosen by trial and error.

### Spoken explanation

The instructor explains that the algorithm cannot run until these choices are made. The number of neighbors \(k\) controls how local the prediction is. A small \(k\), especially \(k=1\), can make the classifier very sensitive to individual examples and outliers. A large \(k\) smooths the decision but may become too global and ignore local structure.

Distance is the other key decision. KNN only makes sense after defining what “near” means. Different data types may require different notions of distance. Numeric features may use Euclidean distance; binary features may need Hamming distance; mixed numeric, binary, and categorical data make the choice harder.

### Verbal examples and asides not on the slide

- The instructor says that in his own practice, Euclidean distance is usually the default unless the data gives a reason not to use it.
- He says he rarely uses Manhattan distance in practice, but it is still legitimate.
- A student asks about \(L_\infty\). The instructor explains that \(L_\infty\) would use the largest coordinate-wise difference as the distance. He says it is an interesting distance, though he cannot immediately think of a common reason to use it in KNN.
- The instructor says \(L_p\) norms will matter later in the course for other machine-learning ideas.

---

## 8. Distance Functions for KNN

### Written slide / notes content

The notes list several distance choices.

### Euclidean distance

Euclidean distance is the standard straight-line distance:

\[
\operatorname{dist}(x,x_i) = \sqrt{(x_1-x_{i1})^2 + (x_2-x_{i2})^2 + \cdots + (x_M-x_{iM})^2}
\]

### \(L\)-norm distances

The notes write the general \(L\)-norm form:

\[
\operatorname{dist}(x,x_i) = \left(|x_1-x_{i1}|^L + |x_2-x_{i2}|^L + \cdots + |x_M-x_{iM}|^L\right)^{1/L}
\]

Special case:

- \(L=1\): Manhattan distance.

### Cosine distance

The notes mention:

\[
\cos(x,x_i)
\]

as a cosine-based distance or similarity.

### Hamming distance

For binary features:

\[
x \in \{0,1\}^M
\]

The notes define Hamming distance as the number of disagreements between two binary vectors.

### Spoken explanation

The September 10 recording begins in the middle of the Hamming-distance example. The instructor is comparing two binary vectors and counting how many positions differ. He initially counts three disagreements, then a student notices another mismatch. The instructor corrects the distance to four.

He uses this to show that even distance definitions, which look simple, require attention. For binary feature vectors, it often does not make sense to use the same intuition as continuous Euclidean geometry. Instead, counting disagreements may be more natural.

The instructor then broadens the point: the distance function depends on the data. If all features are numerical, Euclidean distance may be reasonable. If the features are binary, Hamming distance may be more appropriate. If the dataset mixes binary, categorical, and numerical features, the choice becomes harder and must be thought through.

### Verbal examples and asides not on the slide

- The student correction in the Hamming example is preserved: the instructor says the distance should be four, not three.
- The instructor says this illustrates why even the “simplest” algorithm can be tricky: we still have to decide \(k\), decide the distance, handle binary features, and think about mixed feature types.

---

## 9. Feature Scaling and Data Preprocessing

### Written slide / notes content

The notes ask:

> What if features are of different scales?

Example:

- \(x_1\): size of CPU in millimeters, scale around \(10^{-3}\).
- \(x_2\): speed in GHz, scale around \(10^9\).

A drawing shows that distance becomes dominated by \(x_2\); the data effectively collapses near one axis when plotted on the scale of the large feature.

The notes then give the fix:

> Feature scaling / normalization

Two common transformations are written.

### Min-max scaling

\[
x'_1 = \frac{x_1 - \min(x_1)}{\max(x_1) - \min(x_1)}
\]

This maps the feature to:

\[
x'_1 \in [0,1]
\]

### Standardization

\[
x'_1 = \frac{x_1 - \operatorname{mean}(x_1)}{\operatorname{std}(x_1)}
\]

This gives the transformed feature:

\[
\operatorname{mean}(x'_1)=0, \qquad \operatorname{std}(x'_1)=1
\]

The notes summarize the goal:

> Give all features an equal chance / equal importance.

This is labeled as **data preprocessing**:

> any change/modification of \(D\) before applying ML

### Spoken explanation

The instructor explains that distance-based methods are very sensitive to the scale of features. If one feature is measured in very small units and another in very large units, the large-scale feature can dominate the distance calculation, even if it is not actually more important for prediction.

He uses the CPU example. Suppose one feature is physical CPU size, measured in millimeters, and another is processor speed, measured in gigahertz. Because gigahertz has a scale near \(10^9\), distances will be almost entirely determined by processor speed. The CPU-size feature may become practically useless, even if it was important for the task.

The fix is to rescale features before running KNN. Min-max scaling squeezes every feature into the interval \([0,1]\). Standardization subtracts the mean and divides by the standard deviation, giving a transformed feature with mean zero and standard deviation one.

The instructor frames this as a general machine-learning practice: before applying an algorithm, one may need to modify the dataset. That process is called data preprocessing.

### Verbal examples and asides not on the slide

- The instructor says that when you discover scale problems, you should not “sit down and cry”; you should rescale the data.
- He says some people like standardization because mean and standard deviation connect naturally to probability/statistics.
- He emphasizes that preprocessing is not only for KNN. Many machine-learning algorithms require or benefit from preprocessing.

---

## 10. Measuring the Success of a KNN Classifier: Train/Test Split

### Written slide / notes content

The notes ask:

> How to measure success of KNN classifier?

The first step is written as:

> split the data

The notes show a labeled dataset \(D\), for example with \(N=50\), randomly split into:

- \(D_{train}\): \(2/3\) of the data, e.g. 35 examples.
- \(D_{test}\): \(1/3\) of the data, e.g. 15 examples.

The notes emphasize:

- \(D_{train}\): use it exclusively to create \(f\).
- \(D_{test}\): use it exclusively to estimate the accuracy of \(f\).

A diagram shows:

\[
D \rightarrow D_{train} \rightarrow \text{Create } f
\]

and then:

\[
D_{test} \rightarrow f \rightarrow \text{estimate accuracy}
\]

### Spoken explanation

The instructor explains that before even choosing an accuracy formula, we need the correct evaluation procedure. We must split the data into training and test sets.

The training set is used to create the model. For KNN, this means storing the training examples and using them as the neighbor database. The test set is held out and used only after the model is created, to estimate how well the model works on examples that were not used to build it.

He uses \(N=50\) as a concrete example. If two-thirds are used for training, then about 35 examples are training examples and about 15 are test examples. For each of the 15 test examples, KNN pretends the label is unknown, finds neighbors among the 35 training examples, predicts a label, and then compares the prediction with the true test label.

The instructor strongly warns against evaluating on the same data used to create the model. He says this is one of the worst mistakes in machine learning.

### Verbal examples and asides not on the slide

- The instructor jokes that using the same dataset for training and testing is a reason to go to “machine learning jail.”
- He compares it to an exam: if an exam uses exactly the same questions as the homework, a student may simply memorize answers, so the exam would not measure real understanding. Similarly, a model can memorize training data, so testing on training data gives a misleading estimate.
- He notes the unfortunate tradeoff: if the dataset is small, holding out test data leaves even less data for training.

---

## 11. Confusion Table for Binary Classification

### Written slide / notes content

The notes build a binary-classification evaluation table. For each test example, we compare:

- the true label,
- the predicted label \(f(x)\).

The notes then draw a **2×2 confusion table** with:

- columns: true label \(+\), true label \(-\),
- rows: predicted label \(+\), predicted label \(-\).

The four entries are:

- **TP**: true positive.
- **TN**: true negative.
- **FP**: false positive.
- **FN**: false negative.

### Spoken explanation

The instructor explains that after predicting labels for the test set, we count the types of outcomes. Correct positive predictions are true positives. Correct negative predictions are true negatives. Mistakes are split into false positives and false negatives depending on which direction the mistake goes.

The confusion table is a compact way to represent all binary-classification outcomes. Once we have this table, we can derive several numerical measures of classifier performance.

### Verbal examples and asides not on the slide

- The instructor says some people call this a contingency table, but he prefers the common machine-learning name “confusion table,” because it shows the classifier’s confusion.
- He stresses that TP, TN, FP, and FN are counts, not probabilities.

---

## 12. Accuracy, Error Rate, Precision, Recall, and F1

### Written slide / notes content

The notes list accuracy measures for binary classification.

### Percent accuracy

\[
\text{Accuracy} = \frac{TP + TN}{N_{test}} \cdot 100\%
\]

This is the percent of correct guesses.

### Error rate

\[
\text{Error Rate} = 1 - \frac{TP + TN}{N_{test}}
\]

This is the fraction of errors.

### Precision

\[
P = \frac{TP}{TP + FP}
\]

### Recall

\[
R = \frac{TP}{TP + FN}
\]

### F1-score

\[
F1 = \frac{2PR}{P + R} \in [0,1]
\]

The notes state:

> large F1 is good

### Spoken explanation

The instructor explains percent accuracy as the most natural single-number measure: count correct predictions and divide by the total number of test examples. Error rate is the complementary idea: how often the model is wrong.

Precision and recall focus specifically on the positive class. Precision asks: among the examples predicted positive, how many were actually positive? Recall asks: among the examples that truly were positive, how many did the model successfully catch?

F1 combines precision and recall into one number. The instructor does not dwell on why it has that exact formula, but emphasizes that it is commonly used and that larger values are better.

### Verbal examples and asides not on the slide

- The instructor explains precision and recall by pointing to different parts of the confusion table. Precision uses the predicted-positive row/region; recall uses the true-positive column/region.
- He says many other accuracy measures exist, but these are representative and important.

---

## 13. Cost of Mistakes: Not All Errors Are Equal

### Written slide / notes content

The notes introduce:

> Cost of mistakes — not all mistakes are equal

Example:

> Blood tests for breast cancer diagnosis

The next page contains a **cost matrix**. The matrix has predicted labels as rows and true labels as columns. Correct decisions have cost 0. The example costs are:

- False positive: the model predicts cancer/positive when the true label is negative; example cost around \$10,000.
- False negative: the model predicts negative/no cancer when the true label is positive; example cost around \$1,000,000.

The notes write the general idea of average cost:

\[
\text{Average Cost} = \frac{\text{total cost over test examples}}{N_{test}}
\]

In terms of the confusion table, the total cost is obtained by multiplying each confusion-table count by the cost assigned to that type of outcome.

### Spoken explanation

The instructor explains that percent accuracy may be insufficient because it treats all mistakes equally. In many real problems, different mistakes have very different consequences.

For a blood-test classifier used in cancer screening, a false positive means the test says cancer may be present when it is not. That can cause anxiety, additional tests, and extra cost. A false negative means the test says the person is fine when they actually have cancer. That can delay treatment and may be much more severe medically, socially, and financially.

The class discusses possible dollar amounts. The instructor settles on a false-negative cost around one million dollars and a false-positive cost around ten thousand dollars, not as exact medical economics but as a way to encode the intuition that one mistake is much worse than the other.

A classifier designed with this cost matrix would prefer false alarms over missed cancer cases.

### Verbal examples and asides not on the slide

- A student asks whether the cost is for the hospital or the person. The instructor clarifies that it is a cost assigned for the prediction problem; it may represent consequences to the patient, hospital, society, or whoever defines the objective.
- The instructor says that in this kind of problem, simply saying “95% accurate” is often not enough. We must ask what kinds of mistakes the classifier makes.

---

## 14. Multiclass Classification

### Written slide / notes content

The notes ask:

> What if I have more than 2 classes? Multiclass classification?

Example:

- 3 classes \(\Rightarrow\) create a 3×3 confusion table.

The notes then evaluate which measures extend naturally:

1. Percent accuracy: yes.
2. F1: marked as not directly applicable in the simple binary form.
3. Average cost: yes, if we can define a 3×3 cost matrix.

### Spoken explanation

The instructor explains that the confusion-table idea generalizes easily. If there are three classes, use a 3×3 table. If there are ten classes, use a 10×10 table.

Percent accuracy still works: add the diagonal entries, because those are correct predictions, and divide by the total number of test examples. Off-diagonal entries are mistakes.

The binary precision/recall/F1 formulas do not directly transfer in the same simple form, because TP, FP, FN, and TN were defined for a two-class table. The instructor treats the binary F1 formula as not directly applicable for the multiclass case in this lecture.

Average cost still works if we can define a cost for every pair of predicted and true classes. In a 3×3 setting, this means defining a 3×3 cost matrix and multiplying the count in each cell by its corresponding cost.

### Verbal examples and asides not on the slide

- The instructor says there are many other multiclass evaluation measures used in practice and research, but the lecture focuses on the most representative ones.
- A student correctly points out that a multiclass cost matrix can still assign a separate cost to each kind of mistake.

---

## 15. Live Demo: Iris Dataset and KNN in Python

### Written slide / notes content

This portion is from the spoken lecture and live coding demo rather than the handwritten PDF page.

The instructor uses the classic **Iris dataset**:

- 150 examples.
- 3 classes/species.
- 50 examples per class.
- 4 numerical features.
- Features are measurements of flower parts such as sepal and petal length/width.

The demo uses Python libraries including:

- NumPy,
- plotting tools,
- scikit-learn / `sklearn`, especially `neighbors`,
- the built-in Iris dataset from scikit-learn.

### Spoken explanation

The instructor introduces Iris as a classic benchmark dataset in machine learning. It contains three kinds of iris flowers. They may look similar to non-botanists, but botanists distinguish them as different species. The dataset represents each flower using four numerical measurements.

For visualization, the instructor intentionally keeps only the first two features and discards the other two. He does this not because it is best for prediction, but because two dimensions can be plotted.

He then creates a KNN classifier in scikit-learn. The key machine-learning line is the classifier creation line using the nearest-neighbor algorithm and a chosen value of \(k\). He mentions that most of the surrounding code is not machine learning; it is only for plotting the decision regions.

The plot shows each flower as a dot and the background color as the class that KNN would predict for a new flower at that location. For \(k=1\), each region is controlled by the nearest single training example. A small isolated point can create its own little prediction island.

The instructor changes \(k\) and shows how the decision regions change:

- With \(k=1\), the classifier is very local and sensitive to individual examples.
- With \(k=3\), some tiny isolated regions disappear because the single nearest point no longer controls the vote.
- With \(k=9\), regions become different again, though the behavior may still look strange in some areas.
- With very large \(k\), such as 55 or 149, the classifier becomes very global.
- With \(k=150\), meaning the whole dataset votes, everything becomes the same class, because the global majority dominates.

This motivates the idea that \(k\) must be chosen empirically.

### Verbal examples and asides not on the slide

- The instructor says the plotting code was “stolen from the internet,” not generated by ChatGPT.
- A student asks whether changing \(k\) to 3 means the classifier uses the three nearest neighbors. The instructor confirms this.
- The instructor uses the phrase “if you live there, this is your prediction” to interpret the colored background regions.
- He notes that some regions look unreasonable when \(k\) changes, which reinforces the need for trial and error.

---

## 16. Live Demo: Train/Test Split and Accuracy on Iris

### Written slide / notes content

This is also from the spoken live demo, connected to the handwritten train/test split notes.

The demo performs:

1. Random train/test split.
2. Approximately 66% training and 33% testing.
3. KNN model creation on the training data.
4. Prediction on the test data.
5. Accuracy computation on the test set.

### Spoken explanation

The instructor explains that the earlier decision-region plot was only for intuition. A real experiment requires splitting the dataset into training and test parts.

He uses scikit-learn to randomly split the Iris data so that about one-third becomes the test set. Then he trains the KNN classifier on the training set and computes accuracy on the test set.

He tries \(k=1\) and obtains accuracy values around the mid-70% range in one run. Because the split is random, rerunning the split can give slightly different results. Then he tries \(k=5\), and the accuracy values tend to be somewhat higher in the runs he shows, such as around the low 80% or high 70% range.

The instructor says this is the basic process for choosing \(k\): try different values, repeat the train/test process, and see which value performs better.

### Verbal examples and asides not on the slide

- The instructor says students will see similar code in homework and will play with it themselves.
- He emphasizes that every random split can give a different estimate, so one run is not the whole story.
- He frames this as practical trial and error rather than a purely theoretical choice.

---

## 17. Live Demo: Comparing KNN with Other Classifiers

### Written slide / notes content

This portion is from the spoken demo, not from the handwritten KNN notes.

The instructor shows several two-dimensional toy datasets:

- two moons,
- concentric circles,
- a left/right or more linearly separable dataset.

He compares decision regions from different classifiers.

### Spoken explanation

The instructor shows how different algorithms split feature space.

For KNN, dark regions mean the classifier is very confident because nearby examples all belong to the same class. Lighter regions mean the neighborhood contains a mix of classes, so the vote is less pure.

He then contrasts KNN with a linear classifier. A linear classifier draws a straight line and predicts one class on one side and another class on the other side. This works well for linearly separable or nearly linearly separable data, but it cannot naturally capture moons or circles.

He also points to decision-tree-like behavior, where boundaries are horizontal or vertical and split the space into rectangles because decision trees ask if-else questions about feature values.

Finally, he mentions more advanced algorithms, such as neural networks, which can capture more complex structures like two moons or inner/outer circles.

### Verbal examples and asides not on the slide

- The instructor says the visually appealing algorithms in the middle of the comparison are often the most powerful, and the course will spend time on representatives of these algorithm families.
- He uses this demo to preview future course topics: linear classifiers, decision trees, and neural networks.

---

## 18. Is KNN a Good Supervised-Learning Algorithm?

### Written slide / notes content

The notes ask:

> Is KNN a good S.L. algorithm?

The answer written is:

> Yes, under ideal conditions.

The ideal condition is:

- a lot of data,
- very dense neighborhoods,
- almost infinite number of very near neighbors.

The notes say that if neighborhoods are very dense, then we can estimate:

\[
P(Y \mid X)
\]

very accurately.

But the notes then state:

> Unfortunately, datasets common in ML very rarely have dense neighborhoods.

### Spoken explanation

The instructor says KNN is actually very good under ideal conditions. The reason is intuitive: if every point has many extremely close neighbors, then the labels of those neighbors provide strong evidence about the correct label.

He compares this to living in Manhattan versus living in the middle of Pennsylvania. In Manhattan, there are many people nearby; in the middle of nowhere, the nearest person may still be far away. KNN wants the Manhattan situation: many close neighbors.

Under very dense neighborhoods, the algorithm can estimate the conditional probability of each class given the feature location. For example, if many examples live almost exactly where the new point lives, we can count their labels and estimate class probabilities accurately.

However, the instructor says this ideal condition is rare in machine-learning datasets. Even datasets with many examples often fail to have dense neighborhoods once the number of features is large.

### Verbal examples and asides not on the slide

- The instructor says KNN likes close neighbors because it relies on guilt by association.
- He emphasizes the difference between the closest neighbor and a genuinely near neighbor. If the closest neighbor is still far away, the KNN assumption is weak.

---

## 19. Curse of Dimensionality: Neighborhoods Disappear

### Written slide / notes content

The notes introduce the key fact:

> When \(M\) is large, even if \(N\) is very large, the neighborhood is very sparse.

Equivalently:

> As the number of features grows, the neighborhood disappears.

This is labeled:

> Curse of Dimensionality

The notes then give a geometric example.

Assume a unit square/cube/hypercube with side length 1 and \(N\) examples uniformly distributed inside.

Define the “neighborhood” around a central point as a smaller box whose side length is \(1/2\).

For \(M=2\):

\[
\text{neighbors} = N\left(\frac12\right)^2 = \frac{N}{4}
\]

For \(M=3\):

\[
\text{neighbors} = N\left(\frac12\right)^3 = \frac{N}{8}
\]

For \(M=1\):

\[
\text{neighbors} = N\left(\frac12\right)
\]

For general \(M\) features:

\[
\text{near neighbors} = N\left(\frac12\right)^M
\]

The notes give a high-dimensional example:

\[
N = 1{,}000{,}000 = 10^6, \qquad M = 100
\]

Then:

\[
N\left(\frac12\right)^M
= 10^6 \cdot \frac{1}{2^{100}}
\approx 10^6 \cdot \frac{1}{10^{30}}
= 10^{-24}
\]

So the expected number of near neighbors is essentially zero.

The notes also state:

\[
\frac{\text{distance to nearest neighbor}}{\text{distance to farthest neighbor}} \to 1 \quad \text{as } M \to \infty
\]

### Spoken explanation

The instructor says this is one of the key ideas in machine learning. Our intuition comes from living in two or three dimensions. On a map, we understand what it means for someone to be nearby or far away. Philadelphia has many people within a few miles, while Australia is thousands of miles away.

But in high-dimensional feature spaces, this intuition breaks. If the data lives in 100 dimensions, there may be no genuinely close neighbors at all. Even more strangely, the nearest neighbor and farthest neighbor may be almost the same distance away.

The instructor proves the intuition with simple geometry. In one dimension, a neighborhood of side length one-half contains half the population. In two dimensions, the square neighborhood contains one-fourth. In three dimensions, the cube neighborhood contains one-eighth. In \(M\) dimensions, the fraction is \((1/2)^M\). This shrinks exponentially fast.

With one million examples and 100 features, even a large dataset gives essentially zero expected neighbors inside the half-side-length neighborhood. Therefore, KNN’s basic assumption — that useful near neighbors exist — becomes fragile in high dimensions.

### Verbal examples and asides not on the slide

- The instructor calls the curse of dimensionality an enemy of all machine-learning algorithms, especially nearest neighbors.
- He says 100 features is not unusual or even especially large in machine learning; students will see such datasets in homework and projects.
- He says he will later show a Python example to convince students that in high dimensions the geometry behaves strangely.
- At the end, he apologizes for running over time and says he “owes” the class several minutes.

---

## 20. Consolidated Takeaways from the KNN Lecture

### Written content consolidated from the slides

KNN is a supervised-learning classifier built around a simple rule:

1. Store labeled training examples.
2. For a new example, compute distances to training examples.
3. Select the \(k\) closest examples.
4. Predict by majority vote.

The main design choices are:

- value of \(k\),
- distance function,
- preprocessing/scaling of features,
- evaluation method and accuracy measure.

The main formulas and tools introduced are:

- Euclidean, \(L_p\), cosine, and Hamming distances.
- Train/test split.
- Confusion table: TP, TN, FP, FN.
- Accuracy, error rate, precision, recall, F1.
- Cost matrix and average cost.
- Multiclass confusion table.
- Curse of dimensionality: \(N(1/2)^M\) near-neighbor count.

### Spoken explanation consolidated from the transcript

The instructor’s larger message is that KNN is simple enough to understand quickly, but rich enough to reveal many central machine-learning issues:

- Data comes from sampling, and sampling matters.
- A model must be defined before it can be learned.
- Model quality must be measured with a specific metric.
- Training and testing must be separated.
- Hyperparameters must be chosen before learning.
- The right distance depends on the data type and scale.
- Preprocessing can determine whether a distance-based method works at all.
- Accuracy alone may hide important differences between types of mistakes.
- KNN works best when neighborhoods are dense.
- High-dimensional data destroys naive neighborhood intuition.

KNN therefore serves as the first full supervised-learning algorithm in the course and as a gateway to broader machine-learning concepts.

