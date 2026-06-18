
# === IMPORTS ===
import os
import subprocess as sp
import shutil
import time
import glob
import pandas as pd
import numpy as np
from openbabel import pybel
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier
from rdkit import Chem
from rdkit.Chem import Descriptors

# === FUNCIONES AUXILIARES ===
def read_model_attribute_order(model_file):
    base = os.path.splitext(model_file)[0]
    txt_file = base + ".txt"
    if os.path.exists(txt_file):
        try:
            with open(txt_file, "r") as f:
                line = f.readline().strip()
                order = [x.strip() for x in line.split(",") if x.strip() != ""]
                return order
        except Exception as e:
            print(f"Error leyendo {txt_file}: {e}")
    return None

# === NUEVA FUNCIÓN: Generar descriptores adicionales ===
def generate_additional_descriptors():
    print("Generando descriptores adicionales")
    sdf_path = os.path.join("ToMoCoMD", "chemical_datasets", "to_predict.sdf")
    if not os.path.exists(sdf_path):
        smiles_file = "to_predict.smiles"
        molecules = list(pybel.readfile("smi", smiles_file))
        outfile = pybel.Outputfile("sdf", sdf_path, overwrite=True)
        for mol in molecules:
            mol.addh()
            mol.make3D(forcefield='uff', steps=50)
            mol.localopt(forcefield='uff', steps=2000)
            outfile.write(mol)
        outfile.close()

    descriptors_csv = run_tomocomd(sdf_path)
    base_df = pd.read_csv(descriptors_csv)
    if "molecules" in base_df.columns:
        base_df.drop(columns=["molecules"], inplace=True)

    models_path = "models_electro"
    result_dfs = []

    for model_file in os.listdir(models_path):
        if not model_file.endswith(".model"):
            continue
        model_name = os.path.splitext(model_file)[0]
        txt_path = os.path.join(models_path, f"Headings_{model_name}.txt")
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, "r") as f:
            cols = [x.strip() for x in f.readline().strip().split(",") if x.strip()]
        attr_order = cols[:-1]
        target = cols[-1]

        df_temp = base_df.copy()
        for col in attr_order:
            if col not in df_temp.columns:
                df_temp[col] = np.nan
        df_temp = df_temp[attr_order]
        df_temp[target] = 0
        temp_csv = f"temp_{model_name}.csv"
        df_temp.to_csv(temp_csv, index=False)

        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(temp_csv)
        data.class_index = data.num_attributes - 1

        model_path = os.path.join(models_path, model_file)
        try:
            tupleModel = weka.core.serialization.SerializationHelper.read(model_path)
        except:
            tupleModel = Classifier.deserialize(model_path)
        classifier = tupleModel[0] if isinstance(tupleModel, (list, tuple)) else tupleModel

        predictions = []
        for i in range(data.num_instances):
            inst = data.get_instance(i)
            inst.set_missing(data.class_index)
            pred = classifier.classify_instance(inst)
            predictions.append(pred)

        result_dfs.append(pd.DataFrame({model_name: predictions}))
        os.remove(temp_csv)

    full_df = pd.concat(result_dfs, axis=1)
    full_df.to_csv("descriptores_adicionales.csv", index=False)
    print("descriptores_adicionales.csv generado")

# === ETAPA 1: GENERACIÓN DE ESTRUCTURA Y DESCRIPTORES ===
def create3DSDF_from_smiles(input_smiles):
    print("Generando SDF 3D a partir de SMILES...")
    try:
        molecules = list(pybel.readfile("smi", input_smiles))
    except Exception as e:
        print(f"Error al leer SMILES: {e}")
        exit()
    tomocomd_path = os.path.abspath("ToMoCoMD")
    chem_path = os.path.join(tomocomd_path, "chemical_datasets")
    if not os.path.exists(chem_path):
        os.makedirs(chem_path)
    output_sdf = os.path.join(chem_path, "to_predict.sdf")
    outfile = pybel.Outputfile("sdf", output_sdf, overwrite=True)
    for mol in molecules:
        mol.addh()
        mol.make3D(forcefield='uff', steps=50)
        mol.localopt(forcefield='uff', steps=2000)
        outfile.write(mol)
    outfile.close()
    print(f"Archivo SDF generado: {output_sdf}")
    return output_sdf

def createsmiles_from_3DSDF(input_sdf):
    print("Generando SMILES a partir de SDF 3D...")
    tomocomd_path = os.path.abspath("ToMoCoMD")
    chem_path = os.path.join(tomocomd_path, "chemical_datasets")
    input_sdf = os.path.join(chem_path,input_sdf)
    output_smiles = os.path.abspath("to_predict.smiles")
    with open(output_smiles,'w') as smiles_file:
        try:
            for molecule in pybel.readfile('sdf', input_sdf):
                smiles_file.write(molecule.write('smi'))
        except Exception as e:
            print(f'Error al leer SDF: {e}')
            exit()
    print(f"Archivo SMILE generado: {output_smiles}")
    return output_smiles

def run_tomocomd(sdf_file):
    print("Ejecutando ToMoCoMD...")
    tomocomd_path = os.path.abspath("ToMoCoMD")
    jar_file = os.path.join(tomocomd_path, "ToMoCoMD-CARDD_CLI.jar")
    if not os.path.exists(jar_file):
        print("No se encontró ToMoCoMD-CARDD_CLI.jar en ToMoCoMD.")
        exit()
    cmd = f'cd {tomocomd_path}; java -jar ToMoCoMD-CARDD_CLI.jar'
    status, output = sp.getstatusoutput(cmd)
    print("Salida de ToMoCoMD:")
    print(output)
    search_path = os.path.join(tomocomd_path, "Calculation")
    csv_files = glob.glob(os.path.join(search_path, "**", "*user_specified_headings*.csv"), recursive=True)
    if csv_files:
        descriptors_csv = csv_files[0]
        print(f"Se encontró el CSV de descriptores: {descriptors_csv}")

        # === NUEVA LÓGICA PARA LIMPIAR NaNs ===
        try:
            df = pd.read_csv(descriptors_csv)
            nan_rows = df.isna().any(axis=1).sum()
            if nan_rows > 0:
                print(f"Se encontraron {nan_rows} moléculas con valores NaN. Serán eliminadas.")
                df_clean = df.dropna()
                df_clean.to_csv(descriptors_csv, index=False)
            else:
                print("No se encontraron valores NaN en los descriptores.")
        except Exception as e:
            print(f"Error al revisar/eliminar NaNs en el CSV de descriptores: {e}")
            exit()

        return descriptors_csv
    else:
        print("No se encontró CSV de descriptores. Verifica ToMoCoMD.")
        exit()

def tmcmd_output_mols(descriptors_csv):
    descriptors_csv = os.path.join(os.path.abspath('ToMoCoMD'),'Calculation','headings','to_predict.sdf',descriptors_csv)
    df = pd.read_csv(descriptors_csv)
    mol_names = df.iloc[:,0].tolist()
    corrected_names = []
    for mol_name in mol_names:
        name = mol_name[15:]
        corrected_names.append(name)
    return(corrected_names)

def smile_filterer(filtered_molecules):
    filtered_molecules.append('')
    with open('to_predict.smiles','r') as file:
        textori = file.read()
    filtsmilist = []
    textori = textori.split('\n')
    for oline in textori :
        oline = oline.strip()
        if oline == '':
            continue
        line = oline.split()
        if len(line) == 1:
            line.append('')
        molname = line[1]
        if molname in filtered_molecules :
            filtsmilist.append(oline)
    with open('to_predict.smiles','w') as file:
        for i in filtsmilist:
            file.write(str(i) + '\n')
            
#rdkit ADDITION
def run_rdkit():
    rdkit_file = "rdkit_descriptors.txt"
    smiles_file = "to_predict.smiles"
    output_csv = "output_descriptors.csv"
    run_descriptor_pipeline(rdkit_file, smiles_file, output_csv)
    return output_csv

def run_descriptor_pipeline(rdkit_file, smiles_file, output_csv):
    # Validate files
    for file_path in [rdkit_file, smiles_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"File is empty: {file_path}")

    print("Reading RDKit descriptor list...")
    with open(rdkit_file, "r") as f:
        rdkit_descriptors = [line.strip() for line in f if line.strip()]

    print("RDKit descriptors:", rdkit_descriptors)

    df_smiles = read_smiles_file(smiles_file)
    smiles_list = df_smiles["SMILES"].tolist()

    print("Computing RDKit descriptors...")
    df_rdkit = compute_selected_rdkit_descriptors(smiles_list, rdkit_descriptors)
    print("RDKit descriptor shape:", df_rdkit.shape)

    print("Combining results and saving CSV...")
    df_final = pd.concat([df_smiles, df_rdkit], axis=1)
    print("Final DataFrame shape:", df_final.shape)

    df_final.to_csv(output_csv, index=False)
    print("Descriptors saved to:", output_csv)
    
def compute_selected_rdkit_descriptors(smiles_list, selected_names):
    selected_set = set(name.lower() for name in selected_names)
    selected_funcs = {
        name: func for name, func in Descriptors.descList
        if sanitize_name(name).lower() in selected_set
    }

    if not selected_funcs:
        print("No matching RDKit descriptors found for:", selected_names)

    print(f"Computing {len(selected_funcs)} RDKit descriptors...")

    descriptors_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            desc_vals = {
                sanitize_name(name): func(mol)
                for name, func in selected_funcs.items()
            }
        else:
            desc_vals = {
                sanitize_name(name): np.nan
                for name in selected_funcs
            }
        descriptors_list.append(desc_vals)

    return pd.DataFrame(descriptors_list)

def sanitize_name(name):
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
    
def read_smiles_file(filepath):
    print("Reading SMILES file:", filepath)
    smiles = []
    names = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                smi, name = parts
            elif len(parts) == 1:
                smi = parts[0]
                name = ""
            else:
                continue
            smiles.append(smi)
            names.append(name)
    df = pd.DataFrame({'SMILES': smiles, 'NAME': names})
    print(f"Loaded {len(df)} molecules")
    return df

def load_descriptors(descriptors_csv, additional_csv=None, desired_order=None):
    print("Cargando descriptores desde", descriptors_csv)
    try:
        df = pd.read_csv(descriptors_csv)
        if "molecules" in df.columns:
            df.drop(columns=["molecules"], inplace=True)
        df = df.dropna()
    except Exception as e:
        print(f"Error al cargar descriptores: {e}")
        exit()
    if desired_order is not None:
        if additional_csv and os.path.exists(additional_csv):
            try:
                add_df = pd.read_csv(additional_csv)
                if len(add_df) != len(df):
                    print("El archivo adicional no tiene el mismo número de filas.")
                    exit()
            except Exception as e:
                print("Error al leer archivo adicional:", e)
                exit()
        final_cols = {}
        for col in desired_order:
            if col in df.columns:
                final_cols[col] = df[col]
            else:
                if additional_csv and col in add_df.columns:
                    final_cols[col] = add_df[col]
                else:
                    print(f"La columna {col} no se encontró. Se llenará con NaN.")
                    final_cols[col] = np.nan
        df = pd.DataFrame(final_cols)
        df = df[desired_order]
    return df

# === ETAPA 2: DOMINIO DE APLICABILIDAD CON AMBIT ===
def evaluate_applicability_domain(training_csv, test_csv, model_name, ambit_path):
    print("Evaluando dominio con AMBIT...")
    try:
        training_orig = pd.read_csv(training_csv)
    except Exception as e:
        print(f"Error al leer training original: {e}")
        exit()
    target_header = training_orig.columns[-1]
    originalHeaders = list(training_orig.columns[:-1])
    training = pd.read_csv(training_csv).iloc[:, :-1]
    test = pd.read_csv(test_csv)
    if test.shape[1] == training.shape[1] + 1:
        molecule_names = test.iloc[:, 0].reset_index(drop=True)
        test = test.iloc[:, 1:].reset_index(drop=True)
    else:
        molecule_names = pd.Series([f"Molecule_{i}" for i in range(len(test))], name="Molecule")
    if training.shape[1] != test.shape[1]:
        print("Error: Diferente número de descriptores entre training y test.")
        exit()
    num_desc = training.shape[1]
    standardHeaders = [f'Desc{j}' for j in range(num_desc)]
    training.columns = test.columns = standardHeaders
    training.to_csv(training_csv.replace('.csv', '_AmbitInput.csv'), index=False)
    test.to_csv(test_csv.replace('.csv', '_AmbitInput.csv'), index=False)
    for file in [training_csv.replace('.csv', '_AmbitInput.csv'), test_csv.replace('.csv', '_AmbitInput.csv')]:
        shutil.copy(file, os.path.join(ambit_path, os.path.basename(file)))
    techniques = {
        "DENSITY": {"mode": "_modeDENSITY", "extra": "-r 0.9"},
        "EUCLIDEAN": {"mode": "_modeEUCLIDEAN", "extra": ""},
        "CITYBLOCK": {"mode": "_modeCITYBLOCK", "extra": ""},
        "RANGE": {"mode": "_modeRANGE", "extra": "-r 0.9"}
    }
    domain_flags = []
    for tech_name, params in techniques.items():
        command = (
            f'cd {ambit_path}; java -jar example-ambit-appdomain-jar-with-dependencies.jar '
            f'-m {params["mode"]} -t {os.path.basename(training_csv.replace(".csv", "_AmbitInput.csv"))} '
            f'-s {os.path.basename(test_csv.replace(".csv", "_AmbitInput.csv"))} '
            f'-f {",".join(standardHeaders)} {params["extra"]} '
            f'-o {model_name}_{tech_name}.csv'
        )
        sp.getstatusoutput(command)
        df = pd.read_csv(os.path.join(ambit_path, f'{model_name}_{tech_name}.csv'), on_bad_lines='skip')
        boollist = []
        if tech_name == 'DENSITY' :
            locator = df.iloc[:, -3]
            for i in range(len(locator)) :
                if float(locator.iloc[i]) > 0 :
                    booldata = 0
                elif float(locator.iloc[i]) == 0 :
                    booldata = 1
                boollist.append(booldata)
            flag = pd.DataFrame(boollist, columns = ['DENSITY'])
            print(flag)
        else :
            flag = df.iloc[:, -2]
        domain_flags.append(flag)
    consensus = pd.concat(domain_flags, axis=1)
    consensus['Domain_Count'] = consensus.sum(axis=1)
    in_domain = consensus['Domain_Count'] < 3
    test = test[in_domain].reset_index(drop=True)
    test.columns = originalHeaders
    test.insert(0, "Molecule", molecule_names[in_domain].reset_index(drop=True))
    if target_header not in test.columns:
        test[target_header] = 0
    filtered_csv = test_csv.replace('.csv', '_inDomain.csv')
    test.to_csv(filtered_csv, index=False)
    return filtered_csv, target_header, originalHeaders

# === ETAPA 3: PREDICCIÓN CON MODELOS ENSEMBLE ===
def predict_all_models(test_csv, models_folder, output_predictions_csv, ensemble_output_csv, target_header, originalHeaders):
    model_files = [f for f in os.listdir(models_folder) if f.endswith(".model")]
    ensemble_model = None
    individual_models = []
    for mf in model_files:
        if mf.lower() in ["ensamble.model", "ensemble.model"]:
            ensemble_model = mf
        else:
            individual_models.append(mf)
    test_df = pd.read_csv(test_csv)
    all_preds = {}
    for mf in individual_models:
        model_path = os.path.join(models_folder, mf)
        model_name_attr = os.path.splitext(mf)[0]
        df_temp = test_df.drop(columns=["Molecule"])
        if target_header not in df_temp.columns:
            df_temp[target_header] = 0
        attr_order = read_model_attribute_order(model_path)
        if attr_order is not None:
            df_temp = df_temp[attr_order]
        temp_csv = f"temp_test_{model_name_attr}.csv"
        df_temp.to_csv(temp_csv, index=False)
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(temp_csv)
        data.class_index = data.num_attributes - 1
        try:
            tupleModel = weka.core.serialization.SerializationHelper.read(model_path)
        except:
            tupleModel = Classifier.deserialize(model_path)
        classifier = tupleModel[0] if isinstance(tupleModel, (list, tuple)) else tupleModel
        preds = []
        for i in range(int(data.num_instances)):
            inst = data.get_instance(i)
            inst.set_missing(data.class_index)
            preds.append(classifier.classify_instance(inst))
        all_preds[model_name_attr] = preds
    pred_df = pd.DataFrame({"Molecule": test_df["Molecule"]})
    for col, preds in all_preds.items():
        pred_df[col] = preds
    pred_df.to_csv(output_predictions_csv, index=False)
    numeric_df = pred_df.drop(columns=["Molecule"])
    if ensemble_model is None:
        pred_df[target_header] = numeric_df.mean(axis=1)
    else:
        ensemble_model_path = os.path.join(models_folder, ensemble_model)
        attr_order_ens = read_model_attribute_order(ensemble_model_path)
        df_ens = numeric_df.copy()
        if target_header not in df_ens.columns:
            df_ens[target_header] = 0
        df_ens = df_ens[attr_order_ens] if attr_order_ens is not None else df_ens
        df_ens.to_csv("temp_numeric_predictions.csv", index=False)
        loader = Loader(classname="weka.core.converters.CSVLoader")
        ensemble_data = loader.load_file("temp_numeric_predictions.csv")
        ensemble_data.class_index = ensemble_data.num_attributes - 1
        try:
            tupleModel = weka.core.serialization.SerializationHelper.read(ensemble_model_path)
        except:
            tupleModel = Classifier.deserialize(ensemble_model_path)
        classifier = tupleModel[0] if isinstance(tupleModel, (list, tuple)) else tupleModel
        final_preds = []
        for i in range(int(ensemble_data.num_instances)):
            inst = ensemble_data.get_instance(i)
            inst.set_missing(ensemble_data.class_index)
            final_preds.append(classifier.classify_instance(inst))
        pred_df[target_header] = final_preds
    pred_df[["Molecule", target_header]].to_csv(ensemble_output_csv, index=False)

# === NUEVA FUNCIÓN: LIMPIAR CSVs TEMPORALES ===
# === MODIFICACIÓN TEMPORAL: MANTENER output_descriptors.csv
def clean_temporary_csvs(keep_files=["predicted_ensamble.csv", "model_training.csv",'output_descriptors.csv']):
    print("\nLimpiando archivos CSV temporales...")
    all_csvs = glob.glob("*.csv")
    to_delete = [f for f in all_csvs if f not in keep_files]
    for file in to_delete:
        try:
            os.remove(file)
            print(f"Eliminado: {file}")
        except Exception as e:
            print(f"No se pudo eliminar {file}: {e}")

# === ETAPA 4: MAIN ===
def main():
    print("=== Inicio del proceso ===")
    input_smiles = "to_predict.smiles"
    training_csv = "model_training.csv"
    test_descriptors_csv = "test_descriptors.csv"
    output_predictions_csv = "predicted.csv"
    ensemble_output_csv = "predicted_ensamble.csv"
    models_folder = "models"
    ambit_path = os.path.abspath("Ambit")
    chem_path = os.path.join(os.path.abspath("ToMoCoMD"), "chemical_datasets")
    sdf_preexistente = os.path.join(chem_path, "to_predict.sdf")
    smile_preexistente = os.path.join(os.path.abspath('to_predict.smiles'))

    #generate_additional_descriptors()

    if os.path.exists(sdf_preexistente):
        print(f"Using existing SDF file: {sdf_preexistente}")
        sdf_file = sdf_preexistente
    elif os.path.exists(smile_preexistente):
        sdf_file = create3DSDF_from_smiles(input_smiles)
    else :
        print(r'No SDF nor smile file have been found \nPlease add a smile file in the NuPredictor folder or a SDF file on the NuPredictor->ToMoCoMD->chemical_datasets folder')
        exit()
        
    if os.path.exists(smile_preexistente):
        smiles_file = smile_preexistente
    else: 
        smiles_file = createsmiles_from_3DSDF("to_predict.sdf")
    
    descriptors_csv = run_tomocomd(sdf_file)
    df_descriptors_csv = pd.read_csv(descriptors_csv)
    filtered_molecules = tmcmd_output_mols(descriptors_csv)
    smile_filterer(filtered_molecules)
    #desired_order = pd.read_csv(training_csv).columns[:-1].tolist()
    #descriptor_df = load_descriptors(descriptors_csv, additional_csv="descriptores_adicionales.csv", desired_order=desired_order)
    
    descriptor_rdkit_csv = run_rdkit()
    indexless_rdkit_csv = pd.read_csv(descriptor_rdkit_csv).iloc[:,2:]
    descriptor_df = pd.concat([df_descriptors_csv,indexless_rdkit_csv], axis=1)
    descriptor_df.to_csv(test_descriptors_csv, index=False)

    filtered_test_csv, target_header, originalHeaders = evaluate_applicability_domain(training_csv, test_descriptors_csv, "Model", ambit_path)
    predict_all_models(filtered_test_csv, models_folder, output_predictions_csv, ensemble_output_csv, target_header, originalHeaders)

    print("Proceso de predicción completado.")
    clean_temporary_csvs()

if __name__ == "__main__":
    start_time = time.time()
    jvm.start(system_cp=True, packages=True)
    main()
    jvm.stop()
    print(f"\nTiempo total: {(time.time() - start_time)/60:.2f} minutos.")
