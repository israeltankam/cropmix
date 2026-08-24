# Field geometry and mixture design

## Canonical representation

A `Field` is an arbitrary set of two-dimensional planting coordinates.

```python
field = cm.Field.from_coordinates([
    (0, 0), (1, 0), (2, 0),
    (0.5, 1), (1.5, 1),
])
```

This representation avoids building the epidemiological model around square matrices.

## Convenience constructors

```python
cm.Field.rectangular(rows=20, columns=40, spacing=1.0)
cm.Field.from_mask(mask, spacing=1.0)
cm.Field.from_polygon(boundary, spacing=1.0)
```

For measured fields, supply the actual planting coordinates directly.

## Mixture design

A design is an assignment vector

\[
z=(z_1,\ldots,z_N),
\]

where `z_i` is the variety planted at site `i`.

```python
design = cm.MixtureDesign(
    field,
    ("A", "A", "B", "C", "B"),
)
```

Or generate exact counts randomly:

```python
design = cm.MixtureDesign.random(
    field,
    counts={"A": 40, "B": 35, "C": 25},
    seed=123,
)
```
