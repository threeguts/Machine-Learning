import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from preprocessing import load_imdb_data, preprocess_texts, create_embedding_matrix, load_word2vec_model
import numpy as np
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader, TensorDataset

# ορισμός RNN Model with LSTM
class StackedBiRNN(nn.Module): #ορισμός του μοντέλου ρνν με τις κατάλληλες υπερπαραμέτρους
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, embedding_layer, bidirectional=True): 
        super(StackedBiRNN, self).__init__()

        #ενσωμάτωση λέξεων
        self.embedding = embedding_layer
        #δημιουργία ρνν με χρήση LSTM
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=0.5)
        #οι πιο δημαντικές πληροφορίες
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        #εδώ δημιουργειται το τελικό πλήρως συνδεδεμένο επίπεδο, εφόσον το μοντέλο είναι διπλής κατεύθυνσης: hidden_dim * 2, output_dim 
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        

    def forward(self, x):
    
        if x.dim()==3 and x.shape[1] == 1:  #έλεγχος για να έχουν ίδιες διαστάσεις
            x=x.squeeze(1)
    
        embedded = self.embedding(x).float()
        rnn_out, _ = self.rnn(embedded)  
        pooled = self.global_max_pool(rnn_out.permute(0, 2, 1)).squeeze(2)  
        return self.fc(pooled)

#Word2Vec-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#φόρτωση δεδομένων
x_train, y_train, x_test, y_test = load_imdb_data()

sample_size=1200 #ο υπολογιστής δεν αντέχει παραπάνω
x_train = x_train[:sample_size]
y_train = y_train[:sample_size]
x_test = x_test[:sample_size]
y_test = y_test[:sample_size]

#επεξεργασία κειμένων για φιλτράρισμα του λεξιλογίου
filtered_vocab=preprocess_texts(x_train, x_test)

#φόρτωση του προεκπαιδευμένου μοντέλου Word2Vec
word2vec_model = load_word2vec_model("C:\\Users\\faran\\Desktop\\BiRNN\\GoogleNews-vectors-negative300.bin")

#δημιουργία embedding matrix
embedding_matrix=create_embedding_matrix(filtered_vocab, embedding_dim=300, word2vec_model=word2vec_model)

#μετατροπή του πίνακα σε tensor και δημιουργία του embedding layer
embedding_matrix_tensor = torch.tensor(embedding_matrix)

#δημιουργία του embedding layer χρησιμοποιώντας τις προεκπαιδευμένες ενσωματώσεις λέξεων
embedding_layer = nn.Embedding.from_pretrained(embedding_matrix_tensor, freeze=False) #το freeze=False επιτρέπει την περαιτέρω εκπαίδευση των ενσωματώσεων

#τα δεδομένα μετατρέπονται σε PyTorch tensors και μεταφέρονται στη GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Τraining/Testing-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

x_train = torch.tensor(x_train, dtype=torch.long).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(1)
x_test = torch.tensor(x_test, dtype=torch.long).to(device)#.unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#ορισμός υπερπαραμέτρων μοντέλου
vocab_size=len(filtered_vocab)  #πρέπει το μήκος του λεξιλογίου να είναι ίδιο με το φιλτραρισμένο
embedding_dim=300  #300D Word2Vec embeddings
hidden_dim=64 #ο υπολογιστής δεν αντέχει παραπάνω
num_layers=2
output_dim=1

#αρχικοποίηση μοντέλου
model=StackedBiRNN(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, embedding_layer).to(device)

#loss και optimizer
class_weights = torch.tensor([1.0, 2.0], dtype=torch.float).to(device)  #αύξηση κιλών
criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.3).to(device)) #στάθμιση κιλών
 
optimizer=optim.Adam(model.parameters(), lr=0.0005) #adam optimizer, εκπαιδεύει τον νευρωνικό δίκτυο

#εκπαίδευση με early stopping
epochs=10 #επιλέγουμε 10 εποχές
best_val_loss=float("inf")
patience=5
patience_counter=0

train_losses=[]  #κρατάμε τα training losses
val_losses=[]    #κρατάμε τα validation losses

scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

#για κάθε εποχή
for epoch in range(epochs):
    model.train() #εκπαίδευσε το μοντέλο
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader: #κώδικας ώστε να μην εκραγεί το πσ 
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    #αξιολόγηση στο validation set
    model.eval()
    with torch.no_grad():
        val_predictions = model(x_test).squeeze(1)
        val_loss = criterion(val_predictions, y_test)
    
    # Track validation loss
    val_losses.append(val_loss.item())
    
    print(f"Epoch {epoch+1}: Train Loss = {loss.item()}, Val Loss = {val_loss.item()}")
    scheduler.step(val_loss) 
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping!")
        break

#αξιολόγηση μοντέλου
model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    with torch.no_grad():
        y_pred = model(x_test)  #θρες ν
        y_pred = (torch.sigmoid(y_pred)>0.50).float().cpu().numpy() 

#υπολογισμός Precision, Recall, F1-score
report = classification_report(y_test.cpu().numpy(), y_pred, zero_division=0, digits=4)
print(report)

#γράφημα Loss vs Epochs
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show() #γελάτε γιατι έπεσα κλάψτε γιατί σηκώνομαι