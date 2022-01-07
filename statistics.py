from Bio import Entrez
from Bio import Medline
from Bio import Entrez

pubmed_id = {}
datas = []
def statistics_pubmedid():
    """
    statistics pubmed_id
    """
    with open("D:/DeepLearning/表型数据集/Polymorphisms_and_Phenotypes.txt") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if idx == 0:
                continue
            data = line.strip("\n").split("\t")
            datas.append(data)
            if(len(data[-1])==7 or len(data[-1])==8):
                pubmed_id[data[-1]] = len(pubmed_id)

    print(len(pubmed_id))

def get_abstract(idlist,part):
    """
    obtain part abstract based on pubmed_id

    args:
    idlist : pubmed_id list

    """

    abstractlist = ""

    Entrez.email = 'jiayili@whut.edu.cn'
    hd_efetch = Entrez.efetch(db="pubmed", id=idlist[:part], rettype="medline", retmode="text")
    parse_medline = Medline.parse(hd_efetch)

    for i, ele in enumerate(list(parse_medline)):  # 遍历每一个ID号检索结果
        PMID = ele['PMID']
        abstract = ele['AB']
        abstractlist = abstractlist + abstract + "\n"

    print(abstractlist)

    with open("part_abstract.txt", "w") as f:
        f.write(abstractlist)

if __name__ == "__main__":
    statistics_pubmedid()

    idlist = list(pubmed_id.keys())
    print(type(idlist))

    part = 100
    get_abstract(idlist,part)


