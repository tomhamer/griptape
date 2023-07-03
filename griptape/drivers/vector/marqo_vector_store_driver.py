from typing import Optional
from griptape import utils
from griptape.drivers import BaseVectorStoreDriver
import marqo
from attr import define, field


@define
class MarqoVectorStoreDriver(BaseVectorStoreDriver):
    api_key: str = field(kw_only=True)
    url: str = field(kw_only=True)
    mq: marqo.Client = field(kw_only=True)
    index: str = field(kw_only=True)

    def __attrs_post_init__(self):
        self.mq = marqo.Client(self.url, self.api_key)
        self.mq.index("")

    def upsert_text(
            self,
            string: str,
            vector_id: Optional[str] = None,
            namespace: Optional[str] = None,
            meta: Optional[dict] = None,
            **kwargs
    ) -> str:
        return self.index.add_documents([{"_id": vector_id, namespace: string}])

    def load_entry(self, vector_id: str, namespace: Optional[str] = None) -> Optional[BaseVectorStoreDriver.Entry]:
        result = self.index.fetch(ids=[vector_id], namespace=namespace).to_dict()
        vectors = list(result["vectors"].values())

        if len(vectors) > 0:
            vector = vectors[0]

            return BaseVectorStoreDriver.Entry(
                id=vector["id"],
                meta=vector["metadata"],
                vector=vector["values"],
                namespace=result["namespace"]
            )
        else:
            return None

    def load_entries(self, namespace: Optional[str] = None) -> list[BaseVectorStoreDriver.Entry]:
        # This is a hacky way to query up to 10,000 values from Pinecone. Waiting on an official API for fetching
        # all values from a namespace:
        # https://community.pinecone.io/t/is-there-a-way-to-query-all-the-vectors-and-or-metadata-from-a-namespace/797/5

        results = self.index.query(
            self.embedding_driver.embed_string(""),
            top_k=10000,
            include_metadata=True,
            namespace=namespace
        )

        results = self.index.search(
            "",
            limit=10000
        )

        return [
            BaseVectorStoreDriver.Entry(
                id=r["id"],
                vector=r["values"],
                meta=r["metadata"],
                namespace=results["namespace"]
            )
            for r in results["matches"]
        ]

    def query(
            self,
            query: str,
            count: Optional[int] = None,
            namespace: Optional[str] = None,
            include_vectors: bool = False,
            # PineconeVectorStorageDriver-specific params:
            include_metadata=True,
            **kwargs
    ) -> list[BaseVectorStoreDriver.QueryResult]:

        params = {
            "limit": count if count else BaseVectorStoreDriver.DEFAULT_QUERY_COUNT,
            "attributes_to_retrieve": ["*"] if include_metadata else ["_id"]
        } | kwargs

        results = self.index.search(query, **params)

        if include_vectors:
            results = self.index.get_documents(list(map(lambda x: x["_id"], results)))

        return [
            BaseVectorStoreDriver.QueryResult(
                vector=r["values"],
                score=r["score"],
                meta=r["metadata"],
                namespace=results["namespace"]
            )
            for r in results["hits"]
        ]

    def create_index(self, name: str, **kwargs) -> None:
        mq.create_index(name, index_settings=kwargs)

    def upsert_vector(
            self,
            vector: list[float],
            vector_id: Optional[str] = None,
            namespace: Optional[str] = None,
            meta: Optional[dict] = None,
            **kwargs
    ) -> str:
        raise Exception("not implemented")