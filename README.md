# VulnPropDefense-Model

This repository presents a vulnerability propagation and solves an optimal defense investment problem considering a simple single stage and double stage model.

The files descriptions:
a) vuln_prop_defense_2_nodes_Simple_Model.py : implements the model and solves the problem for a simplistic two node system with a single stage vulnerability propagation.

b) vuln_prop_defense_2_nodes_TwoStage_Model.py : implements the model and solves the problem for a simplistic two node system with a two stage vulnerability propagation.

c) vuln_prop_defense_N_nodes_Simple_Model.py : implements the model and solves the problem for a N node system with a single stage vulnerability propagation.

d) vuln_prop_defense_N_nodes_TwoStage_Model.py : implements the model and solves the problem for a N node system with a two stage vulnerability propagation.

e) Evaluation of ICS network: Texas Synthetic Electric Grid's communication network is a hierarchical model with 3 hierarchical stages: Balancing Authority, Utility COntrol Centers (Market Participants), and their substations.
Hence, the vuln_prop_defense_UtilityControlCenter.py and vuln_prop_defense_Substation.py  are for the utility control centers and substations respectively.

f) The files Utility_0.json (is a sample control center communication network of a utility), while ODESSA_1.json and ODESSA_2.json are two sample substation connected to the Utility.