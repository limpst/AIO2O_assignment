import uuid
from typing import List

# 1. ParentDocumentRetriever를 대체할 클래스 정의
class CustomParentDocumentRetriever:
    def __init__(self, vectorstore, docstore, child_splitter, parent_splitter=None, id_key="doc_id"):
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.child_splitter = child_splitter
        self.parent_splitter = parent_splitter
        self.id_key = id_key

    def add_documents(self, documents: List):
        # 부모 문서 준비
        if self.parent_splitter:
            parent_docs = self.parent_splitter.split_documents(documents)
        else:
            parent_docs = documents

        parents_to_add = []
        children_to_add = []

        for p_doc in parent_docs:
            # 부모 ID 생성
            doc_id = str(uuid.uuid4())
            p_doc.metadata[self.id_key] = doc_id
            parents_to_add.append((doc_id, p_doc))

            # 자식 문서 생성 및 ID 연결
            child_docs = self.child_splitter.split_documents([p_doc])
            for c_doc in child_docs:
                c_doc.metadata[self.id_key] = doc_id
                children_to_add.append(c_doc)

        # 저장소에 저장
        self.docstore.mset(parents_to_add) # docstore가 InMemoryStore인 경우 mset 사용
        self.vectorstore.add_documents(children_to_add)

    def get_relevant_documents(self, query: str):
        # 자식 검색 -> 부모 ID 추출 -> 부모 문서 반환
        sub_docs = self.vectorstore.similarity_search(query)
        parent_ids = list(set([d.metadata[self.id_key] for d in sub_docs if self.id_key in d.metadata]))
        return self.docstore.mget(parent_ids)
