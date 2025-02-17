
import math

import numpy as np
class Stump:

    def __init__(self):
        self.best_feature=None
        self.threshold=None
        self.polarity=None
        self.alpha=None

    def fit(self, examples, w): #εκπαίδευδη του stump

        features_list = []
        label_list = []
        
        for pair in examples:
            features, labels = pair
            features_list.append(features)
            label_list.append(labels)

        samples=len(features_list) #αριθμ΄ός δειγματων
        if samples>0:
            features = len(features_list[0]) #αριθμός χαρακτηριστικ΄ων ανα δείγμα
        else:
            features=0

        min_error=float('inf') #άπειρο σφάλμα για αρχή
        for feature in range(features): #για κάθε χαρακτηριστικό του δείγματος
            feat_values=[features_list[i][feature] for i in range(samples)] #λίστα που περιέχει όλες τις τίμες του συγκεκριμένου χαρακτηριστικού απο όλα τα δειγματα
            
            thresholds = np.sort(np.unique(feat_values)) #όλες οι διαφορετικές τιμές που εμπεριέχονται στα δείγματα του χαρακτηριστικού
            for threshold in thresholds: #για κάθε μια τιμή
                for polarity in [1, -1]:
                    predictions = np.ones(samples) #στην αρχή θέτουμε όλες τις υποθέσεις ως 1
                      
                    if polarity == 1:                           #αν 1, σημαίνει πως προβλέπουμε 1 αν η τιμή είναι>=θ, αλλίως αν είναι μικρότερη , -1
                        predictions[feat_values < threshold]=-1
                    else:
                        predictions[feat_values >= threshold]=-1
                        predictions[feat_values < threshold]=1

                    error = np.sum(w * (predictions != label_list))  #εδώ ελέγχουμε κάτα πόσο οι υποθεσεις μας αντιστοιχούν στις πραγματικές αποκρίσεις

                    if error<min_error:        #ενημέρωση των μεταβλητών μας αν το σφάλμα ειναι σημαντικά μικρό
                        min_error=error
                        self.best_feature=feature
                        self.threshold=threshold
                        self.polarity=polarity    
            
    def predict(self, x):
        feature_value=x[self.best_feature]
        if self.polarity==1:
            if feature_value >= self.threshold:
                return 1 
            else:
                return-1
        else:
            if feature_value >= self.threshold:
                return -1  
            else: 
                return 1
        
            

