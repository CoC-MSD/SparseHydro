import os 

HERE = os.path.dirname(__file__)

CVG_AIR_TEMPERATURE = {
    "PATH": os.path.join(HERE, r"air_temp.csv"),
    "COLUMNS": {
        "Air Temperature": "air_temp_set_1",
    },
    "COMMENT": "#"
}