import math

from rag_core.embeddings import HashingEmbedder


async def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(64)
    a = await embedder.embed_query("Annual billing discount")
    b = (await embedder.embed_documents(["Annual billing discount"]))[0]
    assert a == b
    assert len(a) == 64
    assert math.isclose(sum(v * v for v in a), 1.0, rel_tol=1e-9)


async def test_hashing_embedder_ranks_overlapping_text_higher() -> None:
    embedder = HashingEmbedder(384)
    query = await embedder.embed_query("monthly uptime commitment")
    near, far = await embedder.embed_documents(
        ["Nimbus commits to 99.9% monthly uptime", "Archive storage costs $0.004 per GB-month"]
    )

    def dot(x: list[float], y: list[float]) -> float:
        return sum(i * j for i, j in zip(x, y, strict=True))

    assert dot(query, near) > dot(query, far)
