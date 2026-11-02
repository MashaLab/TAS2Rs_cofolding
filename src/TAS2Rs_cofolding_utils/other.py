import json

human_TAS2Rs = ['TAS2R1', 'TAS2R2', 'TAS2R3', 'TAS2R4', 'TAS2R5', 'TAS2R7', 'TAS2R8', 'TAS2R9', 'TAS2R10', 'TAS2R13', 'TAS2R14', 'TAS2R16', 'TAS2R38', 'TAS2R39',
                'TAS2R40', 'TAS2R41', 'TAS2R42', 'TAS2R43', 'TAS2R44', 'TAS2R45', 'TAS2R46', 'TAS2R47', 'TAS2R48', 'TAS2R49', 'TAS2R50', 'TAS2R60']

def read_json(path_to_file):

    with open(path_to_file, 'r') as inp:
        file_content = json.load(inp)

    return file_content

