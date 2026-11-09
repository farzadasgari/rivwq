from __future__ import annotations


def parse_years(text: str, first_year: int, last_year: int) -> list[int]:
    years: list[int] = []
    for item in text.split(","):
        part = item.strip()
        if not part:
            raise ValueError("empty year item")
        if "-" in part:
            begin_text, end_text = part.split("-", 1)
            begin, end = int(begin_text), int(end_text)
        else:
            begin = end = int(part)
        if begin > end:
            raise ValueError(f"year range runs backwards: {part}")
        for year in range(begin, end + 1):
            if not first_year <= year <= last_year:
                raise ValueError(f"year {year} is outside {first_year}-{last_year}")
            if year not in years:
                years.append(year)
    return years
