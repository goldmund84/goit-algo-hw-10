from __future__ import annotations

import pulp


def maximize_production() -> pulp.LpProblem:
    model = pulp.LpProblem("Drink_Production", pulp.LpMaximize)

    lemonade = pulp.LpVariable("Lemonade", lowBound=0, cat="Integer")
    fruit_juice = pulp.LpVariable("Fruit_Juice", lowBound=0, cat="Integer")

    model += lemonade + fruit_juice, "Total_Products"

    model += 2 * lemonade + 1 * fruit_juice <= 100, "Water"
    model += lemonade <= 50, "Sugar"
    model += lemonade <= 30, "Lemon_Juice"
    model += 2 * fruit_juice <= 40, "Fruit_Puree"

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return model


def summarize_solution(model: pulp.LpProblem) -> dict[str, float]:
    variables = model.variablesDict()
    return {
        "Lemonade": variables["Lemonade"].value(),
        "Fruit_Juice": variables["Fruit_Juice"].value(),
        "Total": pulp.value(model.objective),
    }


if __name__ == "__main__":
    problem = maximize_production()
    result = summarize_solution(problem)
    print("Оптимальний план виробництва:")
    print(f"  Лимонад: {result['Lemonade']:.0f} од.")
    print(f"  Фруктовий сік: {result['Fruit_Juice']:.0f} од.")
    print(f"  Загалом: {result['Total']:.0f} од.")
