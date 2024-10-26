"""
RECUPERACIÓN DE INFORMACIÓN:PRACTICA 1
search.py
Authors: Carlos Giralt and Berta Olano

Program to evaluate information retrieval systems given relevance judgements
Usage: python3 evaluation.py -qrels <qrelsFileName> -results <resultsFileName> -output <outputFileName>

"""

import sys
import matplotlib.pyplot as plt
import numpy as np

class InformationNeed:
    def __init__(self, need_id):
        self.need_id = need_id
        # Diccionario que asocia el documento con su relevancia (0 o 1)
        self.documents = {}

    # Método para añadir un documento a la colección junto con su relevancia
    def add_document(self, doc_id, relevancy):
        self.documents[doc_id] = relevancy

     # Método que filtra y devuelve solo los documentos relevantes
    def get_relevant_documents(self):
        return list({doc_id: rel for doc_id, rel in self.documents.items() if rel == 1})
    
    # Método que devuelve todos los documentos (relevantes o no) asociados a una necesidad 
    # de información
    def get_documents(self):
        return list(self.documents.keys())

class Results:
    def __init__(self):
        # Diccionario que asocia cada necesidad de información con una lista de los
        # documentos recuperados para ella.
        self.information_needs = {}
    
    # Método que dado el identificador de una necesidad de información y el de un documento, 
    # añade a la lista de documentos recuperados para dicha necesidad de información el 
    # documento proporcionado. Si no hay ninguna entrada en el diccionario que coincida 
    # con el identificador de la necesidad de información proporcionado, lo añade.
    def add_result(self, need_id, doc_id):
        if need_id not in self.information_needs:
            self.information_needs[need_id] = []
            
        self.information_needs[need_id].append(doc_id)
    
    # Método que devuelve una lista con todos los documentos recuperados (relevantes o no)
    # para la necesidad de información proporcionada.
    def get_documents_from_infoNeed(self, need_id):
        return list(self.information_needs[need_id])
    
    # Método que devuelve una lista con todos los documentos recuperados relevantes
    # para la necesidad de información proporcionada.
    def get_relevant_documents_from_infoNeed(self, need_id: str, infoNeed: InformationNeed):
        res = []
        for doc in self.get_documents_from_infoNeed(need_id):
            if doc in infoNeed.get_relevant_documents():
                res.append(doc)
        return res
    
    # Método que devuelve True si y sólo si hay documentos recuperados para la
    # necesidad de información proporcionada
    def is_infoNeed_nonempty(self, need_id: str) -> bool:
        # Return True if the list associated with need_id is not empty, False otherwise
        return need_id in self.information_needs and len(self.information_needs[need_id]) > 0

class Evaluation:
    def __init__(self):
        # Diccionario que asocia cada necesidad de información con un objeto InformationNeed
        self.information_needs = {}

    # Método para añadir un juicio de relevancia a una necesidad de información, dado su ID,
    # el del documento y la relevancia del mismo
    def add_judgment(self, information_need_id, document_id, relevancy):
        # Si la necesidad de información no existe, crearla
        if information_need_id not in self.information_needs:
            self.information_needs[information_need_id] = InformationNeed(information_need_id)
        
        # Agregar el documento y su relevancia
        self.information_needs[information_need_id].add_document(document_id, relevancy)

    # Método que, dado el ID de una necesidad de información, devuelve el número de true positives
    # en los resultados obtenidos por un sistema de recuperación de información, considerando
    # hasta el k-ésimo documento recuperado por él.
    def tp(self, info_id: str, results: Results, k: int = None) -> int:
        tp = 0
        if k is None:
            k = len(results.get_documents_from_infoNeed(info_id))
        for docid in results.get_documents_from_infoNeed(info_id)[:k]:
            if docid in self.information_needs[info_id].get_relevant_documents():
                tp += 1
        return tp

    # Método que, dado el ID de una necesidad de información, devuelve el número de false positives
    # en los resultados obtenidos por un sistema de recuperación de información, considerando
    # hasta el k-ésimo documento recuperado por él.
    def fp(self, info_id: str, results: Results, k: int = None) -> int:
        fp=0
        if k is None:
            k = len(results.get_documents_from_infoNeed(info_id))
        for docid in results.get_documents_from_infoNeed(info_id)[:k]:
            if docid not in self.information_needs[info_id].get_relevant_documents():
                fp += 1
        return fp
    
    # Método que, dado el ID de una necesidad de información, devuelve el número de false negatives
    # en los resultados obtenidos por un sistema de recuperación de información, considerando
    # hasta el k-ésimo documento recuperado por él.
    def fn(self, info_id: str, results: Results, k: int = None) -> int:
        fn = 0
        if k is None:
            k = len(results.get_documents_from_infoNeed(info_id))
        for docid in self.information_needs[info_id].get_relevant_documents()[:k]:
            if docid not in results.get_documents_from_infoNeed(info_id):
                fn += 1
        return fn


    # Método que, dado el ID de una necesidad de información, devuelve la métrica de precisión obtenida
    # por un sistema de recuperación de información, considerando hasta el k-ésimo documento recuperado por él.
    def precision(self, info_id: str, results: Results, k: int = None) -> float:
        if results.is_infoNeed_nonempty(info_id):
            return self.tp(info_id, results,k)/(self.tp(info_id, results, k)+self.fp(info_id, results, k))
        else:
            return 0

    
    # Método que, dado el ID de una necesidad de información, devuelve la métrica de "recall" obtenida
    # por un sistema de recuperación de información, considerando hasta el k-ésimo documento recuperado por él.
    def recall(self, info_id, results: Results, at_index = None) -> float:
        if results.is_infoNeed_nonempty(info_id):
            if not at_index:
                at_index = len(results.get_documents_from_infoNeed(infoNeed))
            relevant_docs = self.information_needs[info_id].get_relevant_documents()
            retrieved_docs = results.get_documents_from_infoNeed(info_id)[:at_index]  # Considerar los primeros 'at_index' documentos
            
            relevant_retrieved = 0
            for doc in retrieved_docs:
                if doc in relevant_docs:
                    relevant_retrieved += 1  # Cuenta solo los relevantes recuperados hasta aquí

            # Evitar la división por cero si no hay documentos relevantes
            if len(relevant_docs) == 0:
                return 0.0
            
            # Cálculo del recall
            return relevant_retrieved / len(relevant_docs)
        else:
            return 0

    # Método que, dado el ID de una necesidad de información, devuelve la métrica de evaluación de sistemas "F1" obtenida
    # por un sistema de recuperación de información.
    def f1(self, info_id: str, results: Results) -> float:
        if results.is_infoNeed_nonempty(info_id):
            P = self.precision(info_id, results)
            R = self.recall(info_id, results,len(results.get_documents_from_infoNeed(info_id)))
            print(str(P) + "\n")
            print(str(R) + "\n")
            return (2 * P * R) / (P + R)
        else:
            return 0
    
    # Método que, dado el ID de una necesidad de información, devuelve la métrica de evaluación de sistemas "precisión a 10" 
    # obtenida por un sistema de recuperación de información.
    def prec10(self, info_id: str, results: Results) -> float:
        if results.is_infoNeed_nonempty(info_id):
            if len(results.get_documents_from_infoNeed(info_id)) < 10:
                return self.tp(info_id, Results) / 10
            else:
                return self.precision(info_id,results,10)
        else:
            return 0

    # Método que, dado el ID de una necesidad de información, devuelve la métrica de evaluación de sistemas "precisión promedio" 
    # obtenida por un sistema de recuperación de información.
    def average_precision(self, info_id: str, results: Results):
        if results.is_infoNeed_nonempty(info_id):
            sum_precisions = 0.0
            relevant_docs = self.information_needs[info_id].get_relevant_documents()
            retrieved_docs = results.get_documents_from_infoNeed(info_id)
            num_relevant = len(relevant_docs)
            
            if num_relevant == 0:
                return 0.0

            relevant_retrieved_count = 0  # Contador de documentos relevantes recuperados
            for index, doc in enumerate(retrieved_docs):
            #for index, doc in enumerate(retrieved_docs):
                if doc in relevant_docs:
                    relevant_retrieved_count += 1
                    #print(index)
                    # Calculamos la precisión hasta este punto
                    #precision_at_k = relevant_retrieved_count / (index + 1)
                    precision_at_k = self.precision(info_id,results,index+1)
                    sum_precisions += precision_at_k
            
            return sum_precisions / relevant_retrieved_count
        else:
            return 0

    # Método que, dado el ID de una necesidad de información, devuelve los puntos exhaustividad-precisión
    # que permitirían generar la curva de precisión-exhaustividad.
    def recall_precision(self, info_id: str, results: Results):
        precisions = []
        recalls = []

        # Get retrieved and relevant documents
        retrieved_docs = results.get_documents_from_infoNeed(info_id) if results.is_infoNeed_nonempty(info_id) else []
        relevant_docs = self.information_needs[info_id].get_relevant_documents() if info_id in self.information_needs else []

        # If there are no retrieved or relevant documents, return empty lists
        if not retrieved_docs or not relevant_docs:
            return precisions, recalls

        # Calculate precision and recall for relevant documents
        for index, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                precision = self.precision(info_id, results, index + 1)
                recall = self.recall(info_id, results, index + 1)
                
                precisions.append(precision)
                recalls.append(recall)

        return precisions, recalls


    # Método que, dado el ID de una necesidad de información, devuelve los puntos exhaustividad-precisión
    # interpolados que permitirían generar la curva de precisión-exhaustividad interpolada.
    def recall_precision_interpolated(self, info_id: str, results: Results): 
        precision_points = []
        recall_points = []
        
        # Get retrieved and relevant documents
        retrieved_docs = results.get_documents_from_infoNeed(info_id) if results.is_infoNeed_nonempty(info_id) else []
        relevant_docs = self.information_needs[info_id].get_relevant_documents()

        # If no relevant or retrieved documents, return recall levels with zero precisions
        if not retrieved_docs or not relevant_docs:
            return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.0] * 11

        # Calculate precision and recall for each retrieved document
        for index, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                precision = self.precision(info_id, results, index + 1) 
                recall = self.recall(info_id, results, index + 1)  
                
                precision_points.append(precision)
                recall_points.append(recall)
        
        # Standard recall levels for interpolation
        recall_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Interpolate precision for each recall level
        interpolated_precisions = []
        
        for recall_level in recall_levels:
            max_precision = 0.0
            for r, p in zip(recall_points, precision_points):
                if r >= recall_level:
                    max_precision = max(max_precision, p)
            interpolated_precisions.append(max_precision)

        return recall_levels, interpolated_precisions

        

if __name__ == '__main__':
    i = 1
    infor=False
    while (i < len(sys.argv)):
        if sys.argv[i] == '-qrels':
            qrelsFileName = sys.argv[i+1]
            i = i + 1
        if sys.argv[i] == '-results':
            resultsFileName = sys.argv[i+1]  
            i += 1
        if sys.argv[i] == '-output':
            outputFileName = sys.argv[i+1] 
            i += 1
        i = i + 1

    evaluation =Evaluation()
    #cargar el archivo qrels.txt
    with open(qrelsFileName, 'r') as Queryfile:
        for line in Queryfile:
            if line.strip():
                information_need, document_id, relevancy = line.strip().split('\t')
                relevancy = int(relevancy) 
                print(f"Loading judgment: Need ID: {information_need}, Document ID: {document_id}, Relevancy: {relevancy}")
                evaluation.add_judgment(information_need, document_id, relevancy)
    
    results = Results()
    processed_queries = {}

    with open(resultsFileName, 'r') as Resultsfile:
        for line in Resultsfile:
            if line.strip():
                
                information_need, document_id = line.strip().split('\t')
                
                
                if information_need not in processed_queries:
                    processed_queries[information_need] = 0
                
                
                if processed_queries[information_need] < 45:
                    results.add_result(information_need, document_id)
                    processed_queries[information_need] += 1


    with open(outputFileName, 'w') as Outputfile:
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_prec_at_10 = 0.0
        total_ap = 0.0  # MAP
        all_interpolated_precisions = [0.0] * 11  # Para interpolar en 11 puntos
        count = 1
        num_queries = len(evaluation.information_needs)

        interpolated_precisions = [[0.0 for _ in range(11)] for _ in range(num_queries + 1)]

        for infoNeed in evaluation.information_needs:
            Outputfile.write(f"INFORMATION_NEED {count}\n")
            
            precision = evaluation.precision(infoNeed, results)
            recall = evaluation.recall(infoNeed, results)
            if precision + recall > 0:
                f1 = (2.0 * precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            prec_at_10 = evaluation.prec10(infoNeed, results)
            average_precision = evaluation.average_precision(infoNeed, results)
            
            Outputfile.write(f"precision {precision:.3f}\n")
            Outputfile.write(f"recall {recall:.3f}\n")
            Outputfile.write(f"F1 {f1:.3f}\n")
            Outputfile.write(f"prec@10 {prec_at_10:.3f}\n")
            Outputfile.write(f"average_precision {average_precision:.3f}\n")
            
            # Acumulamos para las métricas totales
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_prec_at_10 += prec_at_10
            total_ap += average_precision
            
            # Recall y Precision
            Outputfile.write("recall_precision\n")
            precisions, recalls = evaluation.recall_precision(infoNeed, results)
            for recall_value, precision_value in zip(recalls, precisions):
                Outputfile.write(f"{recall_value:.3f} {precision_value:.3f}\n")
            
            # Interpolación de precisión y recall
            interpolated_recalls, interpolated_precisions[count-1] = evaluation.recall_precision_interpolated(infoNeed, results)
            Outputfile.write("interpolated_recall_precision\n")
            for recall_value, precision_value in zip(interpolated_recalls, interpolated_precisions[count-1]):
                Outputfile.write(f"{recall_value:.3f} {precision_value:.3f}\n")
            
            Outputfile.write(f"\n")
            
            # Acumulamos las interpolaciones
            for i, precision_value in enumerate(interpolated_precisions[count-1]):
                all_interpolated_precisions[i] += precision_value
            count += 1

        # Calculamos las métricas totales
        total_avg_precision = total_precision / num_queries if num_queries else 0
        total_avg_recall = total_recall / num_queries if num_queries else 0
        total_avg_f1 = ((2.0*total_precision*total_recall)/(total_precision + total_recall))/num_queries if num_queries else 0
        total_avg_prec_at_10 = total_prec_at_10 / num_queries if num_queries else 0
        total_avg_ap = total_ap / num_queries if num_queries else 0
        interpolated_avg_precisions = [p / num_queries for p in all_interpolated_precisions]
        
        # Escribimos las métricas totales en el archivo
        Outputfile.write("\nTOTAL\n")
        Outputfile.write(f"precision {total_avg_precision:.3f}\n")
        Outputfile.write(f"recall {total_avg_recall:.3f}\n")
        Outputfile.write(f"F1 {total_avg_f1:.3f}\n")
        Outputfile.write(f"prec@10 {total_avg_prec_at_10:.3f}\n")
        Outputfile.write(f"MAP {total_avg_ap:.3f}\n")
        
        # Escribimos la interpolación total
        Outputfile.write("interpolated_recall_precision\n")
        for i, precision_value in enumerate(interpolated_avg_precisions):
            Outputfile.write(f"{i/10:.3f} {precision_value:.3f}\n")
        
    interpolated_precisions[num_queries]=interpolated_avg_precisions.copy()

    x = np . linspace (0.0 , 1.0 , 11)
    fig , ax = plt.subplots ()
    for i in range(0,num_queries):
        ax.plot(x, interpolated_precisions[i], label =f'information need  {i+1}')
    ax.plot(x, interpolated_precisions[num_queries], label =f'total')
    ax.set_title('precision - recall curve')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.grid(True, axis='y', linestyle='-', color = 'gray')
    plt.legend(loc ='upper right')
    plt.show ()
