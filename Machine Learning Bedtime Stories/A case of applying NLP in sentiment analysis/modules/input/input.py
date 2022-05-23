#%%
import pdfplumber 
from os.path import join, dirname
from os import listdir

#%%


# get file path in a folder
data_folder_path = join( dirname(dirname(dirname(__file__))), 'data'  )
file_path_list = [ join(data_folder_path,_) for _ in listdir(data_folder_path)]
text_name_list = [ _.split('_')[0] for _ in listdir(data_folder_path)]

# extract full text from pdfs
def extract_full_text_pdf(pdf_path):
    with pdfplumber.open( pdf_path ) as pdf:
        page_list = pdf.pages
        text_list = [ _.extract_text() for _ in page_list ]
        full_text = ' '.join(text_list)
    return(full_text)

# 读入PDF
text_list = [ extract_full_text_pdf(_) for _ in file_path_list]











# lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]

# print(lemmed)
# # lemmatizating 
 
# # calculate indicator 
# text = "Natural language processing is an exciting area."

# text_data_list = [ extract_full_text_pdf(_) for _ in file_path_list]
# #import nltkfrom nltk.tokenize import sent_tokenize, 
# #word_tokenizetext = "Natural language processing is an exciting area."
# #print(sent_tokenize(text)) 
# #print(word_tokenize(text))

