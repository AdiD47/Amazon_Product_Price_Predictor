import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def parse_catalog_content(text: str) -> Dict[str, str]:
    """
    Parse catalog content into structured components.
    """
    if pd.isna(text):
        return {
            'item_name': '',
            'bullet_points': '',
            'description': '',
            'value': np.nan,
            'unit': ''
        }

    item_name = ''
    bullet_points = []
    description = ''
    value = np.nan
    unit = ''

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('Item Name:'):
            item_name = line.replace('Item Name:', '').strip()
        elif line.startswith('Bullet Point'):
            bullet_points.append(line.split(':', 1)[1].strip() if ':' in line else '')
        elif line.startswith('Product Description:'):
            description = line.replace('Product Description:', '').strip()
        elif line.startswith('Value:'):
            try:
                val_str = line.replace('Value:', '').strip()
                if val_str and val_str.lower() not in ['nan', 'none', '']:
                    value = float(val_str)
            except:
                value = np.nan
        elif line.startswith('Unit:'):
            unit = line.replace('Unit:', '').strip()

    return {
        'item_name': item_name,
        'bullet_points': ' '.join(bullet_points),
        'description': description,
        'value': value,
        'unit': unit
    }


def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract text-based features from catalog content.
    """
    parsed = df['catalog_content'].apply(parse_catalog_content)

    df['item_name'] = parsed.apply(lambda x: x['item_name'])
    df['bullet_points'] = parsed.apply(lambda x: x['bullet_points'])
    df['description'] = parsed.apply(lambda x: x['description'])
    df['ipq_value'] = parsed.apply(lambda x: x['value'])
    df['ipq_unit'] = parsed.apply(lambda x: x['unit'])

    df['full_text'] = (
        df['item_name'].fillna('') + ' ' +
        df['bullet_points'].fillna('') + ' ' +
        df['description'].fillna('')
    )

    df['text_length'] = df['full_text'].apply(len)
    df['word_count'] = df['full_text'].apply(lambda x: len(str(x).split()))
    df['item_name_length'] = df['item_name'].apply(len)
    df['description_length'] = df['description'].apply(len)
    df['bullet_points_length'] = df['bullet_points'].apply(len)

    return df


def extract_numeric_features(text: str) -> Dict[str, float]:
    """
    Extract numeric values from text (prices, weights, counts, etc.).
    """
    if pd.isna(text):
        return {
            'num_numbers': 0,
            'max_number': 0,
            'min_number': 0,
            'avg_number': 0,
            'sum_numbers': 0
        }

    numbers = re.findall(r'\d+\.?\d*', str(text))
    numbers = [float(n) for n in numbers if n]

    if not numbers:
        return {
            'num_numbers': 0,
            'max_number': 0,
            'min_number': 0,
            'avg_number': 0,
            'sum_numbers': 0
        }

    return {
        'num_numbers': len(numbers),
        'max_number': max(numbers),
        'min_number': min(numbers),
        'avg_number': np.mean(numbers),
        'sum_numbers': sum(numbers)
    }


def extract_brand_keywords(text: str, top_brands: List[str]) -> Dict[str, int]:
    """
    Check for presence of popular brands in text.
    """
    if pd.isna(text):
        return {f'brand_{brand}': 0 for brand in top_brands}

    text_lower = str(text).lower()
    return {f'brand_{brand}': int(brand.lower() in text_lower) for brand in top_brands}


def extract_category_keywords(text: str) -> Dict[str, int]:
    """
    Extract product category features.
    """
    if pd.isna(text):
        return {
            'is_organic': 0,
            'is_food': 0,
            'is_beverage': 0,
            'is_snack': 0,
            'is_kosher': 0,
            'is_gluten_free': 0,
            'is_vegan': 0,
            'is_gmo_free': 0,
            'is_pack': 0
        }

    text_lower = str(text).lower()

    return {
        'is_organic': int('organic' in text_lower),
        'is_food': int(any(word in text_lower for word in ['food', 'eat', 'meal', 'dish'])),
        'is_beverage': int(any(word in text_lower for word in ['drink', 'beverage', 'juice', 'water', 'coffee', 'tea'])),
        'is_snack': int(any(word in text_lower for word in ['snack', 'candy', 'chocolate', 'chip'])),
        'is_kosher': int('kosher' in text_lower),
        'is_gluten_free': int('gluten free' in text_lower or 'gluten-free' in text_lower),
        'is_vegan': int('vegan' in text_lower),
        'is_gmo_free': int('non-gmo' in text_lower or 'non gmo' in text_lower),
        'is_pack': int(any(word in text_lower for word in ['pack of', 'pack', 'ct', 'count']))
    }


def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    numerator = np.abs(predicted - actual)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2

    mask = denominator != 0
    smape_values = np.zeros_like(numerator)
    smape_values[mask] = numerator[mask] / denominator[mask]

    return np.mean(smape_values)


def preprocess_unit(unit: str) -> str:
    """
    Standardize unit strings.
    """
    if pd.isna(unit) or unit == 'None' or unit == '':
        return 'unknown'

    unit = str(unit).lower().strip()

    unit_mapping = {
        'oz': 'ounce',
        'fl oz': 'fluid_ounce',
        'ounce': 'ounce',
        'fluid ounce': 'fluid_ounce',
        'count': 'count',
        'ct': 'count',
        'pound': 'pound',
        'lb': 'pound',
        'lbs': 'pound',
        'gram': 'gram',
        'g': 'gram',
        'kg': 'kilogram'
    }

    for key, value in unit_mapping.items():
        if key in unit:
            return value

    return 'other'
