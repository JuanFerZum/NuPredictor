# NuPredictor
Noncommercial Python application for Mayr's Nucleophilicity Prediction for Windows & Linux operating systems.
## Requisites
### Software
#### Python Libraries
- Pandas https://pandas.pydata.org/docs/getting_started/install.html
- Numpy https://numpy.org/install/
- OpenBabel https://openbabel.org/docs/UseTheLibrary/PythonInstall.html
- python-weka-wrapper3 https://fracpete.github.io/python-weka-wrapper3/install.html
- Rdkit https://www.rdkit.org/docs/Install.html
##### Quick Libraries Installation for Windows
Python3 must be previously installed from `https://www.python.org/downloads/` if it is not installed already.
- Pandas: Run the command `py -m pip install pandas` on the command window.
- Numpy: Run the command `py -m pip install numpy` on the command window.
- OpenBabel: Run the command `py -m pip install -U openbabel` on the command window.
- python-weka-wrapper3:
  - Install OpenJDK as instructed in https://fracpete.github.io/python-weka-wrapper3/install.html in the section "Prerequisites for all platforms"
  - Install Microsoft Visual C++ Redistributable (X64 version) from https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-supported-redistributable-version
  - Run the commands `py -m pip install setuptools wheel` & `py -m pip install python_weka_wrapper3[plots]` on the command window.
- Rdkit: Run the command `py -m pip install rdkit` on the command window.
#### Applications
- Python3
- ToMoCoMD
- Microsoft Visual C++ v14 Redistributable
- OpenJDK
## Setup (Before the first use) 
1. Clone the repository files or download them as a zip file and extract it.
2. Install Python 3 from `https://www.python.org/downloads/` if it is not installed already.
3. Install all libraries as instructed in the `Python Libraries` section, if they are not installed already.
4. Download `ToMoCoMD QuBiLs-MIDAS Command Line Interface (CLI)` and `Software library` folder from http://tomocomd.com/software/qubils-midas. Extract or copy said files in the ToMoCoMD Folder. Avoid changing or moving the files already present in the ToMoCoMD folder.
5. Reboot the system to finish all the installations.
## Instructions
1. Add the input molecules' file.
    - If the input is a ".sdf" file, place it in the /ToMoCoMD/chemical_datasets directory with the name "to_predict.sdf".
    - If the input is a ".smiles file", place it in the main directory with the name "to_predict.smile" (__only 1 option is required__).
    - There are sample files in the main directory (to_predict.smiles) and the /ToMoCoMD/chemical_datasets directory (to_predict.sdf), __make sure to erase them when adding your own file.__
3. Run the NuPredictor.py program.
4. The molecules' name and predicted nucleophilicity are available in the "predicted_ensamble.csv" file at the main directory.
## Recommended repository structure
```
NuPredictor/
├── Ambit/
│   └── example-ambit-appdomain-jar-with-dependencies.jar
├── models/                        # pre-trained Weka models
│   ├── ensamble.model
│   ├── ensamble.txt
│   ├── N_1.model
│   ├── N_1.txt
│   ├── N_2.model
│   ├── N_2.txt
│   ├── N_3.model
│   ├── N_3.txt
│   ├── N_4.model
│   ├── N_4.txt
│   ├── N_5.model
│   ├── N_5.txt
│   ├── N_6.model
│   ├── N_6.txt
├── ToMoCoMD/
│   ├── chemical_datasets/
│       └── to_predict.sdf         # Input file: Replace with own .sdf file to predict or erase
│   ├── lib/                       # download manually
│   ├── headings.txt
│   ├── tomocomd_qubils.in
│   └── ToMoCoMD-CARDD_CLI.jar     # download manually
├── Nupredictor.py
├── CITATION.cff
├── LICENSE
├── README.md
├── model_training.csv             # training descriptors
├── rdkit_descriptors.txt     
└── to_predict.smiles              # Input file: Replace with own .smiles file to predict or erase
```

## References
- The Pandas Develompent Team. Pandas-Dev/Pandas: Pandas. Zenodo. Zenodo February 2020. https://doi.org/10.5281/zenodo.3509134.
- Harris, C. R.; Millman, K. J.; van der Walt, S. J.; Gommers, R.; Virtanen, P.; Cournapeau, D.; Wieser, E.; Taylor, J.; Berg, S.; Smith, N. J.; Kern, R.; Picus, M.; Hoyer, S.; van Kerkwijk, M. H.; Brett, M.; Haldane, A.; del Río, J. F.; Wiebe, M.; Peterson, P.; Gérard-Marchant, P.; Sheppard, K.; Reddy, T.; Weckesser, W.; Abbasi, H.; Gohlke, C.; Oliphant, T. E. Array Programming with NumPy. Nature. 2020. https://doi.org/10.1038/s41586-020-2649-2.
- O’Boyle, N. M.; Banck, M.; James, C. A.; Morley, C.; Vandermeersch, T.; Hutchison, G. R. Open Babel: An Open Chemical Toolbox. J. Cheminform. 2011, 3 (10). https://doi.org/10.1186/1758-2946-3-33.
- Frank, E.; Hall, M.; Witten, I. The WEKA Workbench. https://waikato.github.io/weka-wiki/downloading_weka/.
- Landrum, G.; Tosco, P.; Kelley, B.; Ric; sriniker; gedeck; Vianello, R.; Schneider, N.; Kawashima, E.; Dalke, A.; N, D.; Cole, B.; Swain, M.; Turk, S.; Cosgrove, D.; Savelyev, A.; Vaucher, A.; Wójcikowski, M.; Jones, G.; Jensen, J. H. Rdkit/Rdkit: 2021_09_2 (Q3 2021) Release. Http://Www.Rdkit.Org/. Zenodo September 2, 2021. https://doi.org/10.5281/zenodo.5589557.
