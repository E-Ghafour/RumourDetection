from sklearn.metrics import (classification_report as C_R,
                            accuracy_score as A_S,
                            precision_score as P_S,
                            recall_score as R_S,
                            f1_score as F1_S,
                            confusion_matrix as C_M)
import seaborn as sns

def report_evaluation(y_pred, y_test,
             classification_report = True,
             accuracy_score = True,
             precision_score = True,
             recall_score = True,
             f1_score = True,
             confusion_matrix = True
            ):
    
    if(classification_report):
        print(C_R(y_pred, y_test))
    
    if(accuracy_score):
        print('Accuracy\t\t\t: %.4f'%A_S(y_pred, y_test, average='weighted'))

    if(precision_score):
        print('Precision\t\t\t: %.4f'%P_S(y_pred, y_test, average='weighted'))

    if(recall_score):
        print('Recall\t\t\t: %.4f'%R_S(y_pred, y_test, average='weighted'))

    if(f1_score):
        print('F1_Score\t\t\t: %.4f'%F1_S(y_pred, y_test, average='weighted'))

    if(confusion_matrix):
        confusion_matrix_report = C_M(y_pred, y_test)
        print('confusion matrix: ')
        sns.heatmap(confusion_matrix_report, annot = True, cmap = 'Blues')