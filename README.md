# NuPredictor
Noncommercial Python application for Mayr's Nucleophilicity Prediction
## Requisites
### Software
__Python Libraries__
- Pandas https://pandas.pydata.org/docs/getting_started/install.html
- Numpy https://numpy.org/install/
- OpenBabel https://openbabel.org/wiki/Category:Installation
- python-weka-wrapper3 https://fracpete.github.io/python-weka-wrapper3/install.html
- Rdkit https://www.rdkit.org/docs/Install.html

__Applications__
- ToMoCoMD 
## Setup
1. Clone the repository files or download them as a zip file and extract it.
2. Download ToMoCoMD QuBiLs-MIDAS Command Line Interface (CLI) and Software library folder from http://tomocomd.com/software/qubils-midas. Extract or copy said files in the ToMoCoMD Folder. Avoid changing or moving the files already present in the ToMoCoMD folder.
## Instructions
1. Add the input molecules' file. If the input is a .sdf file, place it in the /ToMoCoMD/chemical_datasets directory with the name "to_predict.sdf". If the input is an .sdf file, place it in the main directory with the name "to_predict.smile" (__only 1 option is required__). There is a test file in the /ToMoCoMD/chemical_datasets directory, __make sure to erase it when adding your own file.__
2. Run the NuPredictor.py program.
3. The name and predicted nucleophilicity are available in the "predicted_ensamble.csv" file at the main directory.
## References
- The Pandas Develompent Team. Pandas-Dev/Pandas: Pandas. Zenodo. Zenodo February 2020. https://doi.org/10.5281/zenodo.3509134.
- Harris, C. R.; Millman, K. J.; van der Walt, S. J.; Gommers, R.; Virtanen, P.; Cournapeau, D.; Wieser, E.; Taylor, J.; Berg, S.; Smith, N. J.; Kern, R.; Picus, M.; Hoyer, S.; van Kerkwijk, M. H.; Brett, M.; Haldane, A.; del Río, J. F.; Wiebe, M.; Peterson, P.; Gérard-Marchant, P.; Sheppard, K.; Reddy, T.; Weckesser, W.; Abbasi, H.; Gohlke, C.; Oliphant, T. E. Array Programming with NumPy. Nature. 2020. https://doi.org/10.1038/s41586-020-2649-2.
- O’Boyle, N. M.; Banck, M.; James, C. A.; Morley, C.; Vandermeersch, T.; Hutchison, G. R. Open Babel: An Open Chemical Toolbox. J. Cheminform. 2011, 3 (10). https://doi.org/10.1186/1758-2946-3-33.
- Frank, E.; Hall, M.; Witten, I. The WEKA Workbench. https://waikato.github.io/weka-wiki/downloading_weka/.
- Landrum, G.; Tosco, P.; Kelley, B.; Ric; sriniker; gedeck; Vianello, R.; Schneider, N.; Kawashima, E.; Dalke, A.; N, D.; Cole, B.; Swain, M.; Turk, S.; Cosgrove, D.; Savelyev, A.; Vaucher, A.; Wójcikowski, M.; Jones, G.; Jensen, J. H. Rdkit/Rdkit: 2021_09_2 (Q3 2021) Release. Http://Www.Rdkit.Org/. Zenodo September 2, 2021. https://doi.org/10.5281/zenodo.5589557.
