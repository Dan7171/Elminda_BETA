import X_initializer_r1
import models_runner_r1
bna_path = 'first_research/BNA_data.csv'
clinical_path = "first_research/Clinical_data_hebrew.csv"
# Step 1: prepare X for model training
tup = X_initializer_r1.generate_X(bna_path,clinical_path) #initial X df
subjects_X = tup[0]
subject_to_subject_group = tup[1]
# Step 2: RUN MODELS AS YOU WISH
X_lr = subjects_X.copy()
#models_runner_r1.lr(subject_to_subject_group, X_lr, k_select_k_best=10,y_name='change_HDRS-17')
X_knn = subjects_X.copy()
models_runner_r1.knn(subject_to_subject_group, X_knn, k_select_k_best = 20,k_knn = 3,y_name ='end_of_treatment_class_HDRS-17' )
