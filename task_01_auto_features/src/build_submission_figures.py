"""Draw report figures from the published result tables; no model runs."""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

REPORT = Path(__file__).resolve().parents[1] / "report"
TEXT = (REPORT / "SUBMISSION_REPORT_RU.md").read_text(encoding="utf-8")
BLUE, ORANGE, GREEN, GRAY = "#206780", "#d78b49", "#478a79", "#a4b4be"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#253247", "text.color": "#253247",
                     "axes.titleweight": "bold", "axes.titlesize": 12,
                     "savefig.facecolor": "white"})


def table(header):
    lines = TEXT.splitlines()
    start = lines.index(header)
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def number(value):
    return float(value.replace(",", "."))


def save(fig, filename):
    fig.savefig(REPORT / filename, dpi=210, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def workflow():
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.set(xlim=(0, 10.5), ylim=(0, 3.2)); ax.axis("off")
    labels = [("История", "События с t < запроса"),
              ("Признаки", "U / I / D / C\n66 → 206 выражений"),
              ("Отбор", "Внутри обучения\nмаска фиксируется"),
              ("Модель", "CatBoost\nобщие настройки"),
              ("Ранжирование", "Общие кандидаты\nNDCG@20, Recall@20")]
    for i, (title, desc) in enumerate(labels):
        x = 0.05 + i * 2.12
        ax.add_patch(FancyBboxPatch((x, 1.65), 1.9, 1.25,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     facecolor="#edf4f6", edgecolor=BLUE, linewidth=1.1))
        ax.text(x + .95, 2.57, title, ha="center", weight="bold", fontsize=11)
        ax.text(x + .95, 2.04, desc, ha="center", va="center", fontsize=9.6)
        if i < 4:
            ax.annotate("", xy=(x + 2.08, 2.27), xytext=(x + 1.94, 2.27),
                        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.5})
    ax.text(.1, 1.15, "Последовательность проверки", fontsize=11, weight="bold")
    ax.plot([.4, 10], [.65, .65], color=GRAY, lw=2)
    for x, label, color in [(1.5, "P001–P005\nразработка", BLUE),
                            (5.2, "P006\nитоговый тест", ORANGE),
                            (8.8, "P007–P008\nдополнительная валидация", GREEN)]:
        ax.scatter([x], [.65], s=80, color=color, zorder=3)
        ax.text(x, .44, label, ha="center", va="top", fontsize=10)
    save(fig, "submission_workflow_ru.png")


def cohorts():
    rows = table("| Группа | UIDC | Auto A1 | Auto A2 | Union | Число признаков A1/A2 |")
    fig, ax = plt.subplots(figsize=(10, 3.6), layout="constrained")
    x = np.arange(len(rows)); width = .19
    for j, (name, color) in enumerate(zip(["UIDC", "Auto A1", "Auto A2", "Union"],
                                         [BLUE, ORANGE, GREEN, GRAY])):
        values = [number(row[j + 1]) for row in rows]
        bars = ax.bar(x + (j - 1.5) * width, values, width=width, label=name, color=color)
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], fontsize=9, padding=3)
    ax.set_xticks(x, [r[0] for r in rows]); ax.set_ylim(0, .095)
    ax.set_ylabel("NDCG@20"); ax.set_title("Разработка: результат зависит от группы пользователей")
    ax.legend(ncol=4, loc="upper right", frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    save(fig, "submission_cohorts_ru.png")


def final_test():
    rows = table("| Метод | Роль в сравнении | NDCG@20 | Recall@20 |")
    fig, (ax, interval) = plt.subplots(1, 2, figsize=(10.5, 3.5),
                                     gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
    values = [number(r[2]) for r in rows]
    bars = ax.barh(np.arange(len(rows)), values,
                   color=[GRAY, GRAY, BLUE, GREEN, ORANGE, GRAY])
    ax.set_yticks(np.arange(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{v:.5f}" for v in values], padding=4, fontsize=10)
    ax.set_xlim(0, .105); ax.set_xlabel("NDCG@20"); ax.set_title("P006: итоговый временной тест")
    ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
    match = re.search(r"разность NDCG@20 между A2 и UIDC составила \*\*([^*]+)\*\*.*?\*\*\[([^;]+); ([^\]]+)\]", TEXT)
    if not match:
        raise ValueError("Primary interval missing from report")
    delta, low, high = map(number, match.groups())
    interval.axvline(0, color=GRAY, ls="--", lw=1)
    interval.errorbar([delta], [0], xerr=[[delta-low], [high-delta]],
                      fmt="o", color=ORANGE, capsize=8, lw=2)
    interval.set_ylim(-1, 1); interval.set_yticks([]); interval.set_xlim(-.007, .007)
    interval.set_title("Основное сравнение: A2 − UIDC")
    interval.set_xlabel("Разность NDCG@20")
    interval.text(0, .6, f"Δ = {delta:+.6f}\n95% интервал", ha="center", fontsize=11)
    interval.text(0, -.5, f"[{low:+.6f}; {high:+.6f}]\nПреимущество не подтверждено", ha="center", fontsize=10)
    interval.spines["left"].set_visible(False)
    save(fig, "submission_test_ru.png")


def catalog():
    rows = table("| Метод | K100 NDCG@20 | K100 Recall@20 | Каталог NDCG@20 | Каталог Recall@20 |")
    match = re.search(r"число доступных целей выросло с ([\d ]+) до ([\d ]+) из ([\d ]+),", TEXT)
    small, full, total = [int(v.replace(" ", "")) for v in match.groups()]
    fig, (coverage, quality) = plt.subplots(1, 2, figsize=(10.5, 4.5),
                           gridspec_kw={"width_ratios": [.85, 1.55]}, layout="constrained")
    values = [100 * small/total, 100 * full/total]
    bars = coverage.bar([0, 1], values, color=[GRAY, BLUE], width=.55)
    coverage.bar_label(bars, labels=[f"{v:.1f}%\n{n} / {total}" for v, n in zip(values, [small, full])], padding=5, fontsize=11)
    coverage.set_xticks([0, 1], ["K100", "Каталог"]); coverage.set_ylim(0, 108)
    coverage.set_ylabel("Покрытие целей, %"); coverage.set_title("P007: доступность цели")
    coverage.grid(axis="y", alpha=.15); coverage.set_axisbelow(True)
    y = np.arange(len(rows))
    k = np.array([number(r[1]) for r in rows]); f = np.array([number(r[3]) for r in rows])
    quality.hlines(y, np.minimum(k, f), np.maximum(k, f), color=GRAY, lw=2)
    quality.scatter(k, y, color=ORANGE, s=45, label="100 кандидатов", zorder=3)
    quality.scatter(f, y, color=BLUE, s=45, label="Весь доступный каталог", zorder=3)
    quality.set_yticks(y, [r[0] for r in rows], fontsize=10); quality.invert_yaxis()
    quality.set_xlim(0, .085); quality.set_xlabel("NDCG@20"); quality.set_title("Качество тех же моделей")
    quality.set_ylim(len(rows) + 1.2, -.7)
    quality.legend(loc="lower right", frameon=False, fontsize=9)
    quality.grid(axis="x", alpha=.15); quality.set_axisbelow(True)
    save(fig, "submission_catalog_ru.png")


def training():
    rows = table("| Обучение | UIDC NDCG@20 | UIDC Recall@20 | A1 NDCG@20 | A1 Recall@20 |")
    fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
    y = np.arange(len(rows))
    for offset, index, name, color in [(-.18, 1, "UIDC: 66 признаков", BLUE),
                                      (.18, 3, "A1: выбранный поднабор", ORANGE)]:
        values = [number(r[index]) for r in rows]
        bars = ax.barh(y + offset, values, height=.32, label=name, color=color)
        ax.bar_label(bars, labels=[f"{v:.5f}" for v in values], padding=4, fontsize=10)
    ax.set_yticks(y, [r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, .095)
    ax.set_xlabel("NDCG@20"); ax.set_title("P008: состав обучения при фиксированных признаках")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
    save(fig, "submission_training_ru.png")


if __name__ == "__main__":
    for draw in (workflow, cohorts, final_test, catalog, training):
        draw()
    print("Saved 5 figures from the report; no experiments launched.")
