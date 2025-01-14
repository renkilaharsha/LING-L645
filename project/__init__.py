#https://www.bls.gov/oes/current/oes_stru.htm

from re import sub

langauges = ["english", "french", "spanish","dutch", "german"]

language_code ={"english":"en", "french":"fr", "spanish":"es","dutch":"nl", "german":"de"}
models  = ["Xlmr_Bert","M_USE","Multi_Bert","MDistill"]

def camel_case(s):
  s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
  return ''.join([s[0].upper(), s[1:]])

domain  = {11: "Management Occupations",
13 :  "Business and Financial Operations Occupations",
15  :"Computer and Mathematical Occupations",
17 : "Architecture and Engineering Occupations",
19 :  "Life, Physical, and Social Science Occupations",
21 : "Community and Social Service Occupations",
23 : "Legal Occupations",
25 : "Educational Instruction and Library Occupations",
27 : "Arts, Design, Entertainment, Sports, and Media Occupations",
29 : "Healthcare Practitioners and Technical Occupations",
31 : "Healthcare Support Occupations",
33 : "Protective Service Occupations",
35 : "Food Preparation and Serving Related Occupations",
37 : "Building and Grounds Cleaning and Maintenance Occupations",
39 : "Personal Care and Service Occupations",
41 : "Sales and Related Occupations",
43 : "Office and Administrative Support Occupations",
45 : "Farming, Fishing, and Forestry Occupations",
47 :"Construction and Extraction Occupations",
49 : "Installation, Maintenance, and Repair Occupations",
51 :"Production Occupations",
53 : "Transportation and Material Moving Occupations"}
