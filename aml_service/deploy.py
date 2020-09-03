import argparse
import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core import Image
from azureml.core.authentication import AzureCliAuthentication
import json
import os, sys

from azureml.core.model import InferenceConfig

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage

print("In deploy.py")
print("Azure Python SDK version: ", azureml.core.VERSION)

print('Opening eval_info.json...')
eval_filepath = os.path.join('./outputs', 'eval_info.json')

try:
    with open(eval_filepath) as f:
        eval_info = json.load(f)
        print('eval_info.json loaded')
        print(eval_info)
except:
    print("Cannot open: ", eval_filepath)
    print("Exiting...")
    sys.exit(0)

model_name = eval_info["model_name"]
model_version = eval_info["model_version"]
model_path = eval_info["model_path"]
model_acc = eval_info["model_acc"]
deployed_model_acc = eval_info["deployed_model_acc"]
deploy_model = eval_info["deploy_model"]

if deploy_model == False:
    print('Model metric did not meet the metric threshold criteria and will not be deployed!')
    print('Exiting')
    sys.exit(0)

print('Moving forward with deployment...')

parser = argparse.ArgumentParser("deploy")
parser.add_argument("--service_name", type=str, help="service name", dest="service_name", required=True)
parser.add_argument("--aks_name", type=str, help="aks name", dest="aks_name", required=True)
parser.add_argument("--aks_region", type=str, help="aks region", dest="aks_region", required=True)
parser.add_argument("--description", type=str, help="description", dest="description", required=True)
args = parser.parse_args()

print("Argument 1: %s" % args.service_name)
print("Argument 2: %s" % args.aks_name)
print("Argument 3: %s" % args.aks_region)
print("Argument 4: %s" % args.description)

print('creating AzureCliAuthentication...')
cli_auth = AzureCliAuthentication()
print('done creating AzureCliAuthentication!')

print('get workspace...')
ws = Workspace.from_config(auth=cli_auth)
print('done getting workspace!')


aks_name = args.aks_name 
aks_region = args.aks_region
aks_service_name = args.service_name

try:
    service = Webservice(name=aks_service_name, workspace=ws)
    print("Deleting AKS service {}".format(aks_service_name))
    service.delete()
except:
    print("No existing webservice found: ", aks_service_name)

compute_list = ws.compute_targets
aks_target = None
if aks_name in compute_list:
    aks_target = compute_list[aks_name]
    
if aks_target == None:
    print("No AKS found. Creating new Aks: {} and AKS Webservice: {}".format(aks_name, aks_service_name))
    prov_config = AksCompute.provisioning_configuration(location=aks_region, agent_count = 2, cluster_purpose='DevTest', vm_size='STANDARD_D2_v3')
    # Create the cluster
    aks_target = ComputeTarget.create(workspace=ws, name=aks_name, provisioning_configuration=prov_config)
    aks_target.wait_for_completion(show_output=True)
    print(aks_target.provisioning_state)
    print(aks_target.provisioning_errors)
    
print("Creating new webservice")

score_path = 'score_fixed.py'
print('Updating scoring file with the correct model name')
with open('aml_service/score.py') as f:
    data = f.read()
with open('score_fixed.py', "w") as f:
    f.write(data.replace('MODEL-NAME', model_name)) #replace the placeholder MODEL-NAME
    print('score_fixed.py saved')

#Get model
model = Model(ws, model_name)

#Create conda Dependencies
conda_packages = ['numpy==1.19.1', "pip==19.2.3"]
pip_packages = ['azureml-sdk==1.12.0', 'azureml-defaults==1.12.0', 'azureml-monitoring==0.1.0a21' ,'xgboost==1.1.1', 'scikit-learn==0.23.1', 'keras==2.3.1', 'tensorflow==2.0.0']
conda_deps = CondaDependencies.create(conda_packages=conda_packages, pip_packages=pip_packages)
myenv = Environment(name='myenv')
myenv.python.conda_dependencies = conda_deps

inf_config = InferenceConfig(entry_script='score_fixed.py', environment=myenv)

aks_config = AksWebservice.deploy_configuration()

service = Model.deploy(workspace=ws,
                           name=aks_service_name,
                           models=[model],
                           inference_config=inf_config,
                           deployment_config=aks_config,
                           deployment_target=aks_target)

service.wait_for_deployment(show_output=True)
print(service.state)

api_key, _ = service.get_keys()
print("Deployed AKS Webservice: {} \nWebservice Uri: {} \nWebservice API Key: {}".
      format(service.name, service.scoring_uri, api_key))

aks_webservice = {}
aks_webservice["aks_service_name"] = service.name
aks_webservice["aks_service_url"] = service.scoring_uri
aks_webservice["aks_service_api_key"] = api_key
print("AKS Webservice Info")
print(aks_webservice)

print("Saving aks_webservice.json...")
aks_webservice_filepath = os.path.join('./outputs', 'aks_webservice.json')
with open(aks_webservice_filepath, "w") as f:
    json.dump(aks_webservice, f)
print("Done saving aks_webservice.json!")

# Single test data
test_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 2, 5, 6, 4, 3, 1, 34]]
# Call the webservice to make predictions on the test data
prediction = service.run(json.dumps(test_data))
print('Test data prediction: ', prediction)

