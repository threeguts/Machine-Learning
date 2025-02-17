# Machine-Learning
A machine learning project on the IMDB dataset, that includes implemantations with Adaboost, Naive Bayes, Logistic Regression and Bidirectional RNNs.

Η παρούσα εργασία εκπονήθηκε στο πλαίσιο του μαθήματος "Τεχνητή Νοημοσύνη" του χειμερινού εξαμήνου 2024-25 στο Οικονομικό Πανεπιστήμιο Αθηνών.Την εργασία έχουν επιμεληθεί οι φοιτήτριες Αναΐς Φαρχάτ, Ελένη Αντωνιάδη Τσαραμπουλίδη, Στυλιανή Μουμτζή. 

Μέρος Α
  Υλοποιήθηκαν οι παρακάτω αλγόριθμοι για την κατάταξη κειμένων σε δύο κατηγορίες:
    Αφελής ταξινομητής Bayes (πολυμεταβλητή μορφή Bernoulli)
    AdaBoost με Decision Stumps
    Λογιστική Παλινδρόμηση (Logistic Regression)

Μέρος Β
  Συγκρίθηκαν οι παραπάνω αλγόριθμοι με τις αντίστοιχες υλοποιήσεις του Scikit-learn.

Μέρος Γ
  Υλοποιήθηκε και συγκρίθηκε ένα Stacked Bidirectional RNN με LSTM και Global Max Pooling σε PyTorch, χρησιμοποιώντας Word Embeddings.

Δεδομένα
  Χρησιμοποιήθηκε το Large Movie Review Dataset (IMDB dataset) από τον σύνδεσμο:
  https://ai.stanford.edu/~amaas/data/sentiment/

Απαιτήσεις
  Python 3.11
  Βιβλιοθήκες: 
    numpy
    tensorflow
    scikit-learn
    scipy
    matplotlib
Για να εγκαταστήσετε τις απαιτούμενες βιβλιοθήκες, ανοίξτε ένα τερματικό (ή γραμμή εντολών) και εκτελέστε την ακόλουθη εντολή:
    pip install numpy tensorflow scikit-learn scipy matplotlib

