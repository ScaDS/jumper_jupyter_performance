# Templates

Static web assets (HTML, CSS, JS) embedded in the package and served from
Python at runtime.

## Structure

```
templates/
├── <feature>/
│   ├── <variant>/
│   │   ├── <component>/
│   │   │   ├── <component>.html
│   │   │   ├── <component>.css
│   │   │   └── <component>.js
│   │   ├── main.js
│   │   ├── styles.css
│   │   └── <root>.html
│   └── <variant2>/
└── <feature2>/
```

## Rules

1. Every leaf directory is a single self-contained component.
2. Each file in a component directory is named after the directory: `<component>/<component>.{html,css,js}`.
3. HTML, CSS, and JS are kept in separate files — no inline styles or scripts.
4. Files at the `<variant>/` level (`main.js`, `styles.css`, root `.html`) are for orchestration only.
5. Top level groups by UI concept (`visualizer`, `report`); second level by variant (`plotly`, …).
6. Every `<variant>/` directory shipping assets must be registered in `pyproject.toml` under
`[tool.setuptools.package-data]`.
7. Python loaders resolve the template path via `Path(__file__)` with an explicit `.parent` chain up
to the package root.
