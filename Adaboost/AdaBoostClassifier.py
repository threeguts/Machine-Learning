import numpy as np
import Stump as Stump

class AdaBoost:

    def __init__(self, L, M):
        self.L=L #αλγόριθμος μάθησης
        self.M=M #αριθμός υποθέσεων
        self.h=[] #εισαγουμε τις Μ υποθέσεις που μαθαίνουμε
        self.z=[] #εισάγουμε τα βάρη ψήφων των υποθέσεων

    def fit(self, examples):
        N=len(examples)
        w = np.ones(N)/N             #διάνυσμα βαρών που για αρχή όλα τα βάρη είναι ίδια
        for m in range(self.M): #για κάθε υπόθεση
            hm=self.L()         #δημιουργούμε αντικείμενο stump 
            hm.fit(examples, w) #το εκπαιδεύουμε με τα παραδειγματα και τα w τους
            
            error=0             #ορίζουμε το αρχικο σφάλμα ως 0
            
            for j in range(N):  #για όλα τα παραδείγματα της υπόθεσης μας
                x,y= examples[j] 
                
                if hm.predict(x)!=y:    #αν η πρόβλεψη δεν δεν ταυτίζεται με την απόκριση
                    error+=w[j] #αύξησε το βάρος, για να μην επιλεγεί αργότερα
            
            if error>= 0.5:     #πολύ μεγάλο σφάλμα, προσπερνάμε
                continue
            zm = (1 / 2) * np.log((1 - error) / (error + 1e-10)) #υπολογισμός του βάρους ψήφου της υπόθεσης
            for j in range(N):  
                x,y = examples[j]
                if hm.predict(x)== y:   #αν είναι ίδια, τότε μει΄ώνουμε το βάρος της
                    w[j] *= np.exp(-zm)
                else:
                    w[j] *= np.exp(zm)

            w=self.normalize(w) #κάνουμε το άθροισμα των βαρών να ισούται με 1 πάλι
            
            self.h.append(hm) 
            self.z.append(zm)
        #print("Weights of weak classifiers:", self.z)
       
    def normalize(self, w):
        total = np.sum(w)          #άθροισμα βαρών
        if total==0:
            return w
        return w/total   #διαιρούμε το κάθε βάρος με το συνολικό άθροισμα
    
    def weighted_majority(self, x):
        sum=0
        # print(f"\nChecking weighted majority for x={x}:")
        for i in range(len(self.h)):             #για κάθε stump 
            prediction=self.h[i].predict(x)      #πάρε την πρόβλεψη του εκάστοτε stump
            sum+=self.z[i]*prediction            #υπολόγισε το σταθμισμένο sum της πρόβλεψης
            
        if sum>0:
            return 1
        else:
            return 0
        

    def predict(self, x):
        preds = []
        for i in x:
            pred = self.weighted_majority(i)
            preds.append(pred)  
        return preds 

    
    