import cropmix as cm

field = cm.Field.from_polygon(
    [(0, 0), (10, 0), (12, 5), (8, 9), (2, 8), (-1, 4)],
    spacing=1.0,
)
print(f"Generated {field.n_sites} planting sites")
