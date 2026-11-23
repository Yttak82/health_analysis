RANDOM_SEED = 42
ALPHA = 0.05
BP_COLUMN = "systolic_bp"
SEX_COLUMN = ["F", "M"]
SMOKER_COLUMN = ["No", "Yes"]
SIM_GROUP_SIZE = 1000
N_BOOTSTRAP = 50000
BOOTSTRAP_START = 500
BOOTSTRAP_STEP = 1000
N_POWER_SIM = 10000
POWER_SAMPLE_SIZE = [100, 200, 400, 800, 1600, 3200]

EXPECTED_DATATYPES = {
    "id" : "int",
    "age": "int",
    "sex": "str",
    "height": "float",
    "weight": "float",
    "systolic_bp": "float",
    "cholesterol": "float",
    "smoker":"str",
    "disease": "int",
           
}