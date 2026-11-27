"""SQL shared by both adapters. Read ancestors and successors in one statement snapshot."""

from twinctx.contracts import COLS


def family_query(table, *, raw_text=False):
    columns = ", ".join("a." + col for col in COLS)
    if raw_text:
        columns += ", a.text_raw"
    return f"""
        with recursive ancestors(id,supersedes) as (
            select id,supersedes from {table} where id=:seed and kind='memory' and corpus='memory'
            union
            select p.id,p.supersedes from {table} p join ancestors n on p.id=n.supersedes
            where p.kind='memory' and p.corpus='memory'
        ), family(id) as (
            select id from ancestors
            union
            select c.id from {table} c join family n on n.id=c.supersedes
            where c.kind='memory' and c.corpus='memory'
        )
        select {columns} from family f join {table} a on a.id=f.id limit 257"""
