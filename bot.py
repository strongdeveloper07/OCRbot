import os
import logging
import re
import time
from io import BytesIO

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_bytes

from google.cloud import vision
from google.oauth2 import service_account

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    ConversationHandler,
    filters,
)

# ===========================================
# 1) КОНФИГУРАЦИЯ GOOGLE CLOUD VISION CLIENT
# ===========================================
#
# Установите здесь полный путь к вашему JSON-ключу сервисного аккаунта:
# Например: "C:/Users/ВашеИмя/keys/passport-ocr-sa.json"
SERVICE_ACCOUNT_JSON = r"C:\Users\Святослав\Desktop\tg_bot_final\model-palace-445815-q2-8884d44f5ff9.json"

# Загружаем креды и создаём Vision клиента
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_JSON
)
GCV_CLIENT = vision.ImageAnnotatorClient(credentials=credentials)


# ===========================================
# 2) ЛОГИРОВАНИЕ
# ===========================================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ===========================================
# 3) ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЙ
# ===========================================
def enhance_image_quality_pil(image: Image.Image) -> Image.Image:
    """
    Улучшение качества изображения с помощью PIL:
    - резкость
    - контраст
    - яркость
    """
    image = image.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)
    return image


def preprocess_image_cv2(img_np: np.ndarray) -> np.ndarray:
    """
    Комплексная предобработка изображения для OCR с помощью OpenCV:
    - CLAHE (контраст)
    - шумоподавление
    - бинаризация (Otsu)
    """
    # Перевод в LAB и CLAHE на L-канал
    lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Шумоподавление
    img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)

    # Пороговая бинаризация
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


# Предварительные относительные координаты зон на российском паспорте
PASSPORT_ZONES = {
    "series": (0.05, 0.05, 0.40, 0.15),   # область, где обычно лежит серия и номер
    "fio":    (0.05, 0.25, 0.95, 0.45),   # область ФИО (примерно)
    "mrz":    (0.00, 0.85, 1.00, 1.00),   # MRZ (нижняя машиночитаемая зона)
}


def extract_passport_zones(img_np: np.ndarray) -> dict:
    """
    Разбивает паспортное изображение (numpy array) на три зоны:
    'series' – серия/номер,
    'fio' – зона с ФИО,
    'mrz' – машиночитаемая зона внизу
    """
    h, w = img_np.shape[:2]
    zones = {}
    for name, (rel_x1, rel_y1, rel_x2, rel_y2) in PASSPORT_ZONES.items():
        x1 = int(rel_x1 * w)
        y1 = int(rel_y1 * h)
        x2 = int(rel_x2 * w)
        y2 = int(rel_y2 * h)
        zones[name] = img_np[y1:y2, x1:x2]
    return zones


# ===========================================
# 4) GOOGLE VISION OCR ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ===========================================
def ocr_with_google(image_np: np.ndarray) -> str:
    """
    Выполняет всестороннее распознавание текста через Google Cloud Vision:
    - Кодирует numpy array в PNG bytes
    - Посылает запрос text_detection
    - Возвращает объединённый результат из всех обнаруженных блоков
    """
    # Конвертируем numpy array (BGR) обратно в RGB для PIL
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    content = buffer.getvalue()

    image = vision.Image(content=content)
    response = GCV_CLIENT.text_detection(image=image)
    texts = response.text_annotations
    if response.error.message:
        logger.error(f"Vision API error: {response.error.message}")
        return ""

    if not texts:
        return ""

    # Первый элемент texts[0].description – это весь текст целиком
    return texts[0].description or ""


def parse_mrz(mrz_text: str) -> dict:
    """
    Парсит MRZ-текст (две строки длиной ~44 символа) для извлечения:
    - passport_number
    - surname, given_names
    """
    lines = [line.strip().replace(" ", "") for line in mrz_text.splitlines() if len(line.strip()) >= 20]
    if len(lines) < 2:
        return {}

    line1, line2 = lines[0], lines[1]
    passport_number = "".join(ch for ch in line2[:9] if ch.isalnum())

    # MRZ формат: ФАМИЛИЯ<<ИМЯ<ОТЧЕСТВО или ИМЯ<<ОТЧЕСТВО
    surname, given_names = "", ""
    if "<<" in line1:
        parts = line1.split("<<", 1)
        surname = parts[0].replace("<", " ").strip()
        given_names = parts[1].replace("<", " ").strip()
    return {
        "passport_number": passport_number,
        "surname": surname,
        "given_names": given_names,
    }


def extract_mrz_data(zone_mrz: np.ndarray) -> dict:
    """
    Извлекает MRZ-зону, улучшает её бинаризацией и распознаёт текст:
    Возвращает словарь с ключами 'passport_number', 'surname', 'given_names'.
    """
    # Сначала приводим MRZ-зону к ч/б и инвертируем для контраста
    gray = cv2.cvtColor(zone_mrz, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binarized = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    mrz_text = ocr_with_google(binarized)
    return parse_mrz(mrz_text)


def extract_series_number_from_text(text: str) -> str:
    """
    Ищет серию и номер паспорта по нескольким шаблонам:
    - 4 цифры + 6 цифр (с пробелом или без)
    - 'Серия 12 34 Номер 567890' и т. п.
    """
    patterns = [
        r"\b(\d{2}\s?\d{2}\s?\d{6})\b",     # '12 34 567890' или '1234567890'
        r"\b(\d{4}\s?\d{6})\b",             # '1234 567890'
        r"[Сс]ерия\s*[:\-]?\s*(\d{2}\s?\d{2})\s*[Нн]омер\s*[:\-]?\s*(\d{6})",
        r"[Нн]омер\s*[:\-]?\s*(\d{6})",
        r"[Сс]ерия\s*[:\-]?\s*(\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            all_digits = "".join(m.groups())
            all_digits = re.sub(r"\D", "", all_digits)
            if len(all_digits) == 10:
                return f"{all_digits[:4]} {all_digits[4:]}"
    return ""


def extract_fio_from_text(text: str) -> str:
    """
    Пытается извлечь ФИО (Фамилия Имя Отчество) любым из следующих способов:
    1. Шаблон, где перед тремя словами есть слова 'ФАМИЛИЯ' или 'ИМЯ'
    2. Просто три подряд слова из заглавных русских букв длиной ≥3
    3. Берём три самых длинных слова в верхнем регистре, если ничего другого не нашлось
    """
    # Убираем цифры и нежелательные символы
    cleaned = re.sub(r"[^А-ЯЁ\s\-]", " ", text.upper())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()


    # Шаблоны
    pats = [
        r"(?:ФАМИЛИЯ|ИМЯ)[^A-ZА-ЯЁ]*([А-ЯЁ]{3,})\s+([А-ЯЁ]{3,})\s+([А-ЯЁ]{3,})",
        r"\b([А-ЯЁ]{3,})\s+([А-ЯЁ]{3,})\s+([А-ЯЁ]{3,})\b",
        r"\b([А-ЯЁ]{3,}(?:\-[А-ЯЁ]{3,})?)\s+([А-ЯЁ]{3,})\s+([А-ЯЁ]{3,})\b",
    ]
    for pat in pats:
        m = re.search(pat, cleaned)
        if m:
            return f"{m.group(1).capitalize()} {m.group(2).capitalize()} {m.group(3).capitalize()}"

    # Если не найдено — берём три самых длинных слова
    words = re.findall(r"\b[А-ЯЁ]{3,}\b", cleaned)
    if len(words) >= 3:
        top3 = sorted(words, key=lambda w: len(w), reverse=True)[:3]
        return " ".join(w.capitalize() for w in top3)

    return ""


def parse_passport_data(image_np: np.ndarray) -> dict:
    """
    Основная функция: принимает numpy-изображение паспорта (1-я страница или скан),
    возвращает словарь:
        {
            "fio": "...",
            "series_number": "...",
            "passport_number": "...",
            "surname": "...",
            "given_names": "...",
        }
    Использует:
    1) MRZ (приоритетно) → extract_mrz_data
    2) Текст из области 'series' → extract_series_number_from_text
    3) Текст из области 'fio' → extract_fio_from_text
    4) Если чего-то нет, ищет во всём full_text (полный OCR текста)
    """
    results = {}

    # 1) Предобработаем изображение и конвертируем его в numpy
    pil_img = Image.fromarray(image_np[:, :, ::-1])  # конверт BGR→RGB для PIL
    pil_img = enhance_image_quality_pil(pil_img)
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    processed = preprocess_image_cv2(img_bgr)

    # 2) Разбиваем на зоны
    zones = extract_passport_zones(processed)

    # 3) Попробуем MRZ (если получилось) — возвращает «passport_number, surname, given_names»
    try:
        mrz_info = extract_mrz_data(zones["mrz"])
    except Exception as e:
        logger.error(f"Ошибка MRZ OCR: {e}")
        mrz_info = {}

    if mrz_info.get("passport_number"):
        # Если длина >=10, разбиваем на 4+6
        num = mrz_info["passport_number"]
        if len(num) >= 10:
            results["series_number"] = f"{num[:4]} {num[4:10]}"
        results["passport_number"] = num
        if mrz_info.get("surname") and mrz_info.get("given_names"):
            results["fio"] = f"{mrz_info['surname']} {mrz_info['given_names']}"

    # 4) Извлечём ПОСЛЕ MRZ: OCR полных зон «series» и «fio»
    zone_series_text = ocr_with_google(zones["series"])
    zone_fio_text = ocr_with_google(zones["fio"])

    # 5) Серия и номер (если ещё нет)
    if "series_number" not in results:
        sn = extract_series_number_from_text(zone_series_text)
        if not sn:
            # поиск в полном тексте
            full_text = ocr_with_google(processed)
            sn = extract_series_number_from_text(full_text)
        if sn:
            results["series_number"] = sn

    # 6) ФИО (если ещё нет)
    if "fio" not in results:
        fio = extract_fio_from_text(zone_fio_text)
        if not fio:
            full_text = ocr_with_google(processed)
            fio = extract_fio_from_text(full_text)
        if fio:
            results["fio"] = fio

    return results


def extract_passport_data(file_bytes: bytes) -> dict:
    """
    Главная обёртка для JPEG/PNG или PDF (первой страницы):
    1) Конвертирует PDF→PNG (dpi=300)
    2) Выбирает первую страницу (либо сразу PIL-изображение)
    3) Переводит в numpy BGR
    4) Запускает parse_passport_data
    5) Возвращает словарь с найденными полями
    """
    try:
        is_pdf = file_bytes[:4] == b"%PDF"
        pil_images = []
        if is_pdf:
            pil_images = convert_from_bytes(file_bytes, dpi=300, first_page=1, last_page=1)
            if not pil_images:
                return {"error": "Не удалось конвертировать PDF"}
        else:
            pil_images = [Image.open(BytesIO(file_bytes))]


        best = {}
        for pil_img in pil_images:
            pil_img = pil_img.convert("RGB")
            img_np = np.array(pil_img)[:, :, ::-1]  # RGB → BGR для OpenCV
            parsed = parse_passport_data(img_np)
            # Если найдены оба поля (fio + series_number), выходим
            if parsed.get("fio") and parsed.get("series_number"):
                return parsed
            # Иначе сохраняем «наилучший» (с максимальным числом ключей)
            if len(parsed) > len(best):
                best = parsed

        return best

    except Exception as e:
        logger.error(f"Ошибка обработки паспорта: {e}", exc_info=True)
        return {"error": f"Ошибка OCR: {e}"}


# ===========================================
# 5) СОСТОЯНИЯ БОТА И КЛАВИАТУРЫ
# ===========================================
(
    COMPANY_NAME,
    COMPANY_INN,
    SELECT_SERVICE,
    SELECT_STAGE,
    UPLOAD_PASSPORT,
    UPLOAD_MIGRATION_CARD,
    UPLOAD_PATENT,
    UPLOAD_DACTYLO,
) = range(8)

SERVICE_OPTIONS = [
    ["Постановка на учет"],
    ["Заключение трудового договора"],
    ["Расторжение трудового договора"],
    ["Снятие с учета"],
    ["Уведомление от работника иностранного гражданина"],
]

STAGE_OPTIONS = [["Первичная"], ["Продление"]]
SKIP_OPTION = [["Пропустить"]]


# ===========================================
# 6) ХЭНДЛЕРЫ ДЛЯ TELEGRAM БОТА
# ===========================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "👋 Добро пожаловать в EasyMigrateBot!\n\n"
        "Введите название вашей компании:",
        reply_markup=ReplyKeyboardRemove(),
    )
    return COMPANY_NAME


async def get_company_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["company_name"] = update.message.text.strip()
    await update.message.reply_text("📝 Введите ИНН вашей компании (10 или 12 цифр):")
    return COMPANY_INN


async def get_company_inn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    inn = update.message.text.strip()
    if not inn.isdigit() or len(inn) not in (10, 12):
        await update.message.reply_text("❌ Неверный ИНН. Введите 10 или 12 цифр:")
        return COMPANY_INN

    context.user_data["company_inn"] = inn
    await update.message.reply_text(
        "🔍 Выберите тип услуги:",
        reply_markup=ReplyKeyboardMarkup(SERVICE_OPTIONS, resize_keyboard=True),
    )
    return SELECT_SERVICE


async def select_service(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["service"] = update.message.text.strip()
    await update.message.reply_text(
        "📅 Выберите этап оформления:",
        reply_markup=ReplyKeyboardMarkup(STAGE_OPTIONS, resize_keyboard=True),
    )
    return SELECT_STAGE


async def select_stage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["stage"] = update.message.text.strip()
    await update.message.reply_text(
        "🛂 Загрузите страницу паспорта с фото и личными данными:\n\n"
        "• Сфотографируйте при хорошем освещении\n"
        "• Избегайте бликов и теней\n"
        "• Держите камеру параллельно документу\n"
        "• Или загрузите PDF-файл\n\n"
        "📌 Совет: для лучшего распознавания:\n"
        "1. Положите документ на тёмную поверхность\n"
        "2. Убедитесь, что все данные читаемы",
        reply_markup=ReplyKeyboardRemove(),
    )
    return UPLOAD_PASSPORT


async def upload_passport(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.document:
        file_obj = await update.message.document.get_file()
    elif update.message.photo:
        file_obj = await update.message.photo[-1].get_file()
    else:
        await update.message.reply_text("❌ Загрузите изображение или PDF:")
        return UPLOAD_PASSPORT

    try:
        await update.message.reply_text("⏳ Скачиваю документ...")
        file_bytes = await file_obj.download_as_bytearray()
        file_bytes = bytes(file_bytes)


        await update.message.reply_text("🔍 Обрабатываю паспорт через Google Vision OCR...")
        start_time = time.time()
        result = extract_passport_data(file_bytes)
        duration = time.time() - start_time
        logger.info(f"OCR completed in {duration:.2f}s")

        if "error" in result:
            await update.message.reply_text(
                f"❌ Ошибка OCR: {result['error']}\n\n"
                "Рекомендации:\n"
                "• Переснимите документ при лучшем освещении\n"
                "• Загрузите PDF вместо фото\n"
                "• Убедитесь, что текст на документе читается\n\n"
                "Попробуйте загрузить заново:"
            )
            return UPLOAD_PASSPORT

        context.user_data["passport_data"] = result

        # Формируем ответ для пользователя
        fio = result.get("fio", "Не найдено")
        series_number = result.get("series_number", "Не найдено")
        response = "✅ Паспорт обработан!\n\n"
        response += f"👤 <b>ФИО:</b> {fio}\n"
        response += f"🔢 <b>Серия и номер:</b> {series_number}\n\n"
        response += "Теперь загрузите миграционную карту:"

        await update.message.reply_text(response, parse_mode="HTML")
        return UPLOAD_MIGRATION_CARD

    except Exception as e:
        logger.error(f"Ошибка обработки паспорта: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Что-то пошло не так при OCR. Попробуйте другой файл или переснимите документ:"
        )
        return UPLOAD_PASSPORT


async def upload_migration_card(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not (update.message.document or update.message.photo):
        await update.message.reply_text("❌ Загрузите изображение или файл:")
        return UPLOAD_MIGRATION_CARD

    await update.message.reply_text("📄 Миграционная карта получена.")
    await update.message.reply_text(
        "📝 Загрузите патент (или нажмите 'Пропустить'):",
        reply_markup=ReplyKeyboardMarkup(SKIP_OPTION, resize_keyboard=True),
    )
    return UPLOAD_PATENT


async def upload_patent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text and "пропустить" in update.message.text.lower():
        await update.message.reply_text("⏭️ Патент пропущён.")
    elif update.message.document or update.message.photo:
        await update.message.reply_text("📄 Патент получен.")
    else:
        await update.message.reply_text("❌ Загрузите файл или нажмите 'Пропустить':")
        return UPLOAD_PATENT

    await update.message.reply_text(
        "🖐 Загрузите дактилоскопическую карту (или нажмите 'Пропустить'):",
        reply_markup=ReplyKeyboardMarkup(SKIP_OPTION, resize_keyboard=True),
    )
    return UPLOAD_DACTYLO


async def upload_dactylo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text and "пропустить" in update.message.text.lower():
        await update.message.reply_text("⏭️ Дактилоскопия пропущена.")
    elif update.message.document or update.message.photo:
        await update.message.reply_text("📄 Дактилоскопическая карта получена.")
    else:
        await update.message.reply_text("❌ Загрузите файл или нажмите 'Пропустить':")
        return UPLOAD_DACTYLO

    # Финальный отчёт
    pd = context.user_data.get("passport_data", {})
    report = (
        "🎉 <b>Все документы загружены!</b>\n\n"
        "<b>Компания:</b> {company}\n"
        "<b>ИНН:</b> {inn}\n"
        "<b>Услуга:</b> {service}\n"
        "<b>Этап:</b> {stage}\n\n"
        "<b>Паспортные данные:</b>\n"
        "• <b>ФИО:</b> {fio}\n"
        "• <b>Серия/номер:</b> {series}\n"
    ).format(
        company=context.user_data.get("company_name", "Не указано"),
        inn=context.user_data.get("company_inn", "Не указан"),
        service=context.user_data.get("service", "Не выбрана"),
        stage=context.user_data.get("stage", "Не выбран"),
        fio=pd.get("fio", "Не распознано"),
        series=pd.get("series_number", "Не распознано"),
    )


    await update.message.reply_text(report, parse_mode="HTML")
    await update.message.reply_text(
        "Для нового оформления введите /start", reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Отменено. Для начала введите /start")
    return ConversationHandler.END


# ===========================================
# 7) ЗАПУСК БОТА
# ===========================================
def main():
    if not os.path.isfile(SERVICE_ACCOUNT_JSON):
        logger.error(
            "JSON-ключ сервисного аккаунта не найден. Проверьте SERVICE_ACCOUNT_JSON."
        )
        return

    app = ApplicationBuilder().token(os.getenv("TELEGRAM_TOKEN")).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            COMPANY_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_company_name)],
            COMPANY_INN: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_company_inn)],
            SELECT_SERVICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_service)],
            SELECT_STAGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_stage)],
            UPLOAD_PASSPORT: [MessageHandler(filters.Document.ALL | filters.PHOTO, upload_passport)],
            UPLOAD_MIGRATION_CARD: [MessageHandler(filters.Document.ALL | filters.PHOTO, upload_migration_card)],
            UPLOAD_PATENT: [
                MessageHandler(filters.TEXT & filters.Regex(r'^(Пропустить|пропустить)$'), upload_patent),
                MessageHandler(filters.Document.ALL | filters.PHOTO, upload_patent),
            ],
            UPLOAD_DACTYLO: [
                MessageHandler(filters.TEXT & filters.Regex(r'^(Пропустить|пропустить)$'), upload_dactylo),
                MessageHandler(filters.Document.ALL | filters.PHOTO, upload_dactylo),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(conv)
    # Дополнительная команда, если пользователь введёт /passport в любой момент
    app.add_handler(CommandHandler("passport", upload_passport))

    app.add_error_handler(lambda update, context: logger.error(f"Ошибка: {context.error}", exc_info=True))

    logger.info("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()
