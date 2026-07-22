#############################################################################################

######################################LangcHAIN COMMUNITY Text Reader########################

#############################################################################################

from langchain_community.document_loaders import TextLoader

file=TextLoader("AI.txt",encoding="utf-8")

docs=file.load()

print(docs)

print(docs[0].page_content)

print(docs[0].metadata)

                


#############################################################################################

######################################LangcHAIN COMMUNITY PDFReader##########################

#############################################################################################



from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("AI_Healthcare_Guide.pdf")

docs=loader.load()


print(docs)


print(docs[0].page_content)

print(docs[0].metadata)


#############################################################################################

######################################LangcHAIN COMMUNITY CSVReader##########################

#############################################################################################


from langchain_community.document_loaders import CSVLoader

loader=CSVLoader("Healthcare_Patients.csv")

docs = loader.load()

print(len(docs))


print(docs[0].metadata)

print(docs[0].page_content)



#############################################################################################

######################################LangcHAIN COMMUNITY URL################################

#############################################################################################


from langchain_community.document_loaders import WebBaseLoader

url="https://en.wikipedia.org/wiki/Artificial_intelligence"

loader=WebBaseLoader(url)

docs=loader.load()

print(docs[0].page_content)

print(docs[0].metadata)


#############################################################################################