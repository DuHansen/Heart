import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from fpdf import FPDF
import seaborn as sns
import warnings
from pathlib import Path
from PIL import Image
import io

# Configurações iniciais
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
sns.set_style("whitegrid")

def preprocess_data(file_path):
    """Pré-processamento dos dados"""
    try:
        df = pd.read_csv(file_path)
        
        print("\n=== Primeiras linhas do dataset ===")
        print(df.head())
        
        print("\n=== Informações do dataset ===")
        print(df.info())

        # Tratamento de valores zerados inválidos
        zero_to_nan_cols = ['restingBP', 'cholesterol', 'restingHR']
        df[zero_to_nan_cols] = df[zero_to_nan_cols].replace(0, np.nan)

        # Preenchimento de valores ausentes
        print("\n=== Valores ausentes antes do tratamento ===")
        missing_before = df.isnull().sum().to_string()
        print(missing_before)

        # Preencher numéricos com mediana
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

        # Preencher categóricos com moda
        cat_cols = df.select_dtypes(exclude=np.number).columns
        for col in cat_cols:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)

        print("\n=== Valores ausentes após tratamento ===")
        missing_after = df.isnull().sum().to_string()
        print(missing_after)

        return df, missing_before, missing_after

    except Exception as e:
        print(f"\nErro no pré-processamento: {str(e)}")
        raise

def prepare_model_data(df):
    """Preparação dos dados para modelagem"""
    try:
        X = df.drop(columns=["heartDisease"])
        y = df["heartDisease"].astype(int)

        categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[ 
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])

        return X, y, preprocessor

    except Exception as e:
        print(f"\nErro na preparação dos dados: {str(e)}")
        raise

def train_and_evaluate(X, y, preprocessor):
    """Treinamento e avaliação do modelo"""
    try:
        pipeline = Pipeline([ 
            ('preprocessor', preprocessor), 
            ('classifier', KNeighborsClassifier()) 
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        print("\n=== Métricas de Desempenho ===")
        print(f"Acurácia: {acc:.4f}")
        print(f"Precisão: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nMatriz de Confusão:")
        print(cm)
        print("\nRelatório de Classificação:")
        print(cr)

        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        print("\n=== Validação Cruzada ===")
        print(f"Acurácias: {cv_scores}")
        print(f"Média: {cv_scores.mean():.4f}")
        print(f"Desvio padrão: {cv_scores.std():.4f}")

        metrics = {
            "Acurácia": f"{acc:.4f}",
            "Precisão": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
            "F1-Score": f"{f1:.4f}"
        }

        cv_results = {
            "scores": str(cv_scores.round(4)),
            "mean": cv_scores.mean(),
            "std": cv_scores.std()
        }

        return pipeline, cm, metrics, cr, cv_results

    except Exception as e:
        print(f"\nErro no treinamento: {str(e)}")
        raise

def generate_visualizations(df, cm, metrics):
    """Geração de todas as visualizações"""
    try:
        img_dir = Path("visualizations")
        img_dir.mkdir(exist_ok=True)
        
        vis_df = df.copy()
        
        # Conversões de colunas categóricas
        vis_df['sex'] = vis_df['sex'].map({'male': 0, 'female': 1})
        vis_df['painType'] = vis_df['painType'].map({
            'Typical Angina': 0, 'Atypical Angina': 1,
            'Non-Anginal Pain': 2, 'Asymptomatic': 3
        })
        vis_df['highBloodSugar'] = vis_df['highBloodSugar'].map({
            True: 1, False: 0, 'yes': 1, 'no': 0
        }).astype(int)
        vis_df['restingECG'] = vis_df['restingECG'].map({
            'Normal': 0, 'ST': 1, 'LVH': 2, 'Hypertrophy': 2
        })
        vis_df['exerciseAngina'] = vis_df['exerciseAngina'].map({
            False: 0, True: 1, 'no': 0, 'yes': 1
        }).astype(int)
        vis_df['STslope'] = vis_df['STslope'].map({
            'Up': 0, 'Flat': 1, 'Down': 2,
            'Upsloping': 0, 'Downsloping': 2
        })
        vis_df['defectType'] = vis_df['defectType'].map({
            'Normal': 0, 'FixedDefect': 1, 'ReversibleDefect': 2
        }).fillna(0)
        vis_df['heartDisease'] = vis_df['heartDisease'].astype(int)

        image_paths = {}

        # 1. Gráfico de métricas
        metrics_path = img_dir / "metrics.png"
        plt.figure(figsize=(10, 5))
        values = [float(metrics['Acurácia']), float(metrics['Precisão']), 
                 float(metrics['Recall']), float(metrics['F1-Score'])]
        labels = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
        colors = sns.color_palette("husl", 4)
        
        bars = plt.bar(labels, values, color=colors)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.title('Métricas de Desempenho')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths['metrics'] = metrics_path

        # 2. Matriz de Confusão
        confusion_path = img_dir / "confusion_matrix.png"
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negativo', 'Positivo'],
                   yticklabels=['Negativo', 'Positivo'])
        plt.title('Matriz de Confusão')
        plt.xlabel('Previsto')
        plt.ylabel('Real')
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths['confusion_matrix'] = confusion_path

        # 3. Matriz de Correlação
        correlation_path = img_dir / "correlation_matrix.png"
        plt.figure(figsize=(12, 10))
        corr_matrix = vis_df.corr(method='pearson')
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Matriz de Correlação de Pearson')
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths['correlation_matrix'] = correlation_path

        # 4. Frequência Cardíaca vs Doença Cardíaca
        hr_path = img_dir / "heart_rate_vs_disease.png"
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='heartDisease', y='restingHR', data=vis_df)
        plt.title('Frequência Cardíaca vs Doença Cardíaca')
        plt.xlabel('Doença Cardíaca (0 = Não, 1 = Sim)')
        plt.ylabel('Frequência Cardíaca')
        plt.savefig(hr_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths['heart_rate'] = hr_path

        # 5. Distribuição de Idade
        age_path = img_dir / "age_distribution.png"
        plt.figure(figsize=(8, 6))
        sns.histplot(vis_df['age'], bins=20, kde=True)
        plt.title('Distribuição de Idade')
        plt.xlabel('Idade')
        plt.ylabel('Frequência')
        plt.savefig(age_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths['age_distribution'] = age_path

        # 6. ST Depression vs Doença Cardíaca (se existir)
        if 'STdepression' in vis_df.columns:
            st_path = img_dir / "st_depression_vs_disease.png"
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='heartDisease', y='STdepression', data=vis_df)
            plt.title('ST Depression vs Doença Cardíaca')
            plt.xlabel('Doença Cardíaca (0 = Não, 1 = Sim)')
            plt.ylabel('ST Depression')
            plt.savefig(st_path, dpi=300, bbox_inches='tight')
            plt.close()
            image_paths['st_depression'] = st_path

        # 7. Distribuição da Pressão Arterial
        bp_path = img_dir / "blood_pressure_distribution.png"
        plt.figure(figsize=(8, 6))
        sns.histplot(vis_df['restingBP'], bins=20, kde=True)
        plt.title('Distribuição da Pressão Arterial')
        plt.xlabel('Pressão Arterial')
        plt.ylabel('Contagem')
        plt.savefig(bp_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths['blood_pressure'] = bp_path

        # 8. Idade vs Colesterol
        if 'cholesterol' in vis_df.columns:
            age_chol_path = img_dir / "age_vs_cholesterol.png"
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x='age', y='cholesterol', data=vis_df, hue='heartDisease', palette='coolwarm')
            plt.title('Idade vs Colesterol')
            plt.xlabel('Idade')
            plt.ylabel('Colesterol')
            plt.savefig(age_chol_path, dpi=300, bbox_inches='tight')
            plt.close()
            image_paths['age_cholesterol'] = age_chol_path

        # 9. Distribuição por restingECG e doença
        resting_ecg_path = img_dir / "resting_ecg_disease.png"
        resting_df = df.copy()
        resting_df['heartDisease'] = resting_df['heartDisease'].astype(bool)
        grouped = resting_df.groupby(['restingECG', 'heartDisease']).size().unstack(fill_value=0)
        grouped.columns = ['Sem Doença', 'Com Doença']
        ax = grouped.plot(kind='bar', figsize=(9, 6), color=['skyblue', 'salmon'])
        plt.title('Distribuição por Tipo de ECG e Doença Cardíaca')
        plt.xlabel('Tipo de ECG')
        plt.ylabel('Quantidade')
        plt.xticks(rotation=0)
        plt.legend(title='Doença Cardíaca')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(resting_ecg_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths['resting_ecg_disease'] = resting_ecg_path

        # 10. Distribuição por faixa etária
        age_dist_path = img_dir / "age_distribution_count.png"
        bins = [20, 30, 40, 50, 60, 70, 80]
        labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
        vis_df['age_group'] = pd.cut(vis_df['age'], bins=bins, labels=labels, right=False)
        age_counts = vis_df['age_group'].value_counts().sort_index()
        ax = sns.barplot(x=age_counts.index, y=age_counts.values, palette="viridis")
        plt.title('Distribuição por Faixa Etária')
        plt.xlabel('Faixa Etária')
        plt.ylabel('Quantidade')
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10),
                        textcoords='offset points', fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(age_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths['age_distribution_count'] = age_dist_path

        # 11. Distribuição por faixa etária e doença
        age_disease_path = img_dir / "age_disease_distribution.png"
        grouped = vis_df.groupby(['age_group', 'heartDisease']).size().unstack()
        grouped.columns = ['Sem Doença', 'Com Doença']
        grouped.plot(kind='bar', stacked=True, color=['lightblue', 'salmon'], figsize=(12, 6))
        plt.title('Distribuição por Faixa Etária e Doença Cardíaca')
        plt.xlabel('Faixa Etária')
        plt.ylabel('Quantidade')
        plt.legend(title='Doença Cardíaca')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(age_disease_path, dpi=300, bbox_inches='tight')
        plt.close()
        image_paths['age_disease_distribution'] = age_disease_path

        return image_paths

    except Exception as e:
        print(f"\nErro ao gerar visualizações: {str(e)}")
        raise

def generate_pdf_report(
    image_paths,
    df_head,
    df_info,
    missing_before,
    missing_after,
    metrics,
    cm,
    cr,
    cv_results
):
    """Geração do relatório em PDF completo"""
    try:
        # Verificar imagens disponíveis
        available_images = {k: v for k, v in image_paths.items() if v.exists()}
        if not available_images:
            raise FileNotFoundError("Nenhuma imagem disponível para gerar o PDF")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Página 1: Informações do dataset
        pdf.add_page()
        pdf.set_font("Arial", size=16, style='B')
        pdf.cell(200, 10, txt="Relatório Completo de Análise Cardíaca", ln=True, align='C')
        
        # Primeiras linhas
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.multi_cell(0, 10, "=== Primeiras linhas do dataset ===")
        pdf.set_font("Courier", size=8)
        with pd.option_context('display.max_columns', None):
            pdf.multi_cell(0, 5, df_head.to_string(index=False))
        
        # Informações do dataset
        pdf.set_font("Arial", size=12)
        pdf.ln(5)
        pdf.multi_cell(0, 10, "=== Informações do dataset ===")
        pdf.set_font("Courier", size=8)
        pdf.multi_cell(0, 5, str(df_info))
        
        # Valores ausentes
        pdf.set_font("Arial", size=12)
        pdf.ln(5)
        pdf.multi_cell(0, 10, "=== Valores ausentes antes do tratamento ===")
        pdf.set_font("Courier", size=8)
        pdf.multi_cell(0, 5, missing_before)
        
        pdf.set_font("Arial", size=12)
        pdf.ln(5)
        pdf.multi_cell(0, 10, "=== Valores ausentes após tratamento ===")
        pdf.set_font("Courier", size=8)
        pdf.multi_cell(0, 5, missing_after)
        
        # Métricas de desempenho
        pdf.set_font("Arial", size=12)
        pdf.ln(5)
        pdf.multi_cell(0, 10, "=== Métricas de Desempenho ===")
        pdf.set_font("Courier", size=10)
        for k, v in metrics.items():
            pdf.cell(0, 6, f"{k}: {v}", ln=True)
        
        # Matriz de Confusão
        pdf.ln(4)
        pdf.multi_cell(0, 10, "Matriz de Confusão:")
        pdf.set_font("Courier", size=8)
        pdf.multi_cell(0, 5, str(cm))
        
        # Relatório de Classificação
        pdf.set_font("Arial", size=12)
        pdf.ln(4)
        pdf.multi_cell(0, 10, "Relatório de Classificação:")
        pdf.set_font("Courier", size=8)
        pdf.multi_cell(0, 5, cr)
        
        # Validação cruzada
        pdf.set_font("Arial", size=12)
        pdf.ln(4)
        pdf.multi_cell(0, 10, "=== Validação Cruzada ===")
        pdf.set_font("Courier", size=10)
        pdf.cell(0, 6, f"Acurácias: {cv_results['scores']}", ln=True)
        pdf.cell(0, 6, f"Média: {cv_results['mean']:.4f}", ln=True)
        pdf.cell(0, 6, f"Desvio padrão: {cv_results['std']:.4f}", ln=True)
        
        # Adicionar gráficos
        for name, path in available_images.items():
            pdf.add_page()
            pdf.set_font("Arial", size=14, style='B')
            pdf.cell(200, 10, txt=f"{name.replace('_', ' ').title()}", ln=True, align='C')
            
            # Ajustar tamanho da imagem
            with Image.open(path) as img:
                img_width, img_height = img.size
                aspect_ratio = img_width / img_height
                new_width = 180
                new_height = new_width / aspect_ratio
                
                # Verificar se a imagem cabe na página
                if new_height > 240:
                    new_height = 200
                    new_width = new_height * aspect_ratio
            
            pdf.image(str(path), x=(210 - new_width)/2, y=30, w=new_width, h=new_height)
        
        # Salvar PDF
        pdf_path = Path("relatorio_cardiaco_completo.pdf")
        pdf.output(str(pdf_path))
        print(f"\nRelatório gerado com sucesso: {pdf_path}")
        
        return pdf_path

    except Exception as e:
        print(f"\nErro ao gerar PDF: {str(e)}")
        raise

def main():
    """Fluxo principal de execução"""
    try:
        file_path = '4.heart.csv'
        
        # Pré-processamento
        df, missing_before, missing_after = preprocess_data(file_path)
        df_head = df.head()
        df_info = str(df.info())
        
        # Preparação e modelagem
        X, y, preprocessor = prepare_model_data(df)
        pipeline, cm, metrics, cr, cv_results = train_and_evaluate(X, y, preprocessor)
        
        # Visualizações
        image_paths = generate_visualizations(df, cm, metrics)
        
        # Gerar PDF
        generate_pdf_report(
            image_paths=image_paths,
            df_head=df_head,
            df_info=df_info,
            missing_before=missing_before,
            missing_after=missing_after,
            metrics=metrics,
            cm=cm,
            cr=cr,
            cv_results=cv_results
        )
        
    except Exception as e:
        print(f"\nErro no processo principal: {str(e)}")

if __name__ == "__main__":
    main()