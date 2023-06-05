import string
import unicodedata

from zemberek import (TurkishMorphology, TurkishSentenceExtractor,
                      TurkishSentenceNormalizer)

remove_punc = str.maketrans('', '', string.punctuation)

second_hand_terms = ["için satıyorum", "icin satiyorum", "hediye olarak geldi", "hediye geldi", "hediye alındı", "hediye alindi"
'kullanılmadı', "kullanilmadi", "kullanmadım", "kullanmadim", "kullandık", "kullandik",
"kullanılmamıştır", "kullanilmamistir", "kullandım", "kullandim," "giyilmedi", "giyildi", "giydim", "giydik", "giyilmiş", "giyilmis"
"giyilmemiştir", "giyilmemistir", 'giymedim', 'giymediler', "giyemedim", "giyemedik",
'çizik var', 'cizik var', 'çiziği var', 'cizigi var', 'yırtık var', "soluk var", "yırtık yoktur",
"bedeni olmadı", "bana olmuyor", "bedeni uymuyor", "küçük geldi", 'kucuk geldi', "küçük oldu", 'kucuk oldu',
"büyük geldi", "buyuk geldi", "büyük oldu", "buyuk oldu", "hatasızdır", "hatasizdir", "sorunsuzdur", "temizdir", "defosuzdur",
"iyi durumda", "sorunu yok", "az kullanıldı", "az kullanildi" "sıfır gibi", "sifir gibi", "sıfır ayarında", "sifir ayarinda",
"mağazasından alınmıştır", "magazasindan alinmistir"
"oyuncudan alınmıştır", "oyuncudan alinmistir", "futbolcudan", "maç sonu", "dolayi satiyorum", "dolayı satıyorum", 
"pazarlık", "indirim", "son fiyat"]

risky_terms = ["tüm bedenler mevcuttur", "tum bedenler mevcuttur", "her bedeni mevcuttur",
 "s m l", "s / m / l", "yeni & etiketli", "etiketli", "s-m-l-xl", "s m xl xxl", "beden seçenekleri", "bedenler",
 "hızlı kargo", "replika", "smlxlxxl", "numaralar mevcuttur", "numaralar vardır", "seçenekleri vardır", "secenekleri vardir",
 "ilan actiriniz", "ilan açtırınız", "ilanı satın almayın", "ilani satin almayin", "s,m,l", "41,42,43,44", "41 42 43 44"]

SPIECE_UNDERLINE = u"▁".encode("utf-8")


class TextNormalization:
    """Text normalization task
    """

    def __init__(self):
        """Constructor method
        """
        self.zemberek_morpholgy = TurkishMorphology.create_with_defaults()
        self.zemberek_normalizer = TurkishSentenceNormalizer(self.zemberek_morpholgy)
        self.zemberek_extractor = TurkishSentenceExtractor()

    def normalize(self,
                  text: str,
                  remove_space: bool = True,
                  do_lower_case: bool = True,
                  normalize_function: str = "NFKC",
                  is_turkish: bool = True,
                  use_zemberek: bool = True):
        """
        Preprocess text by removing extra space and normalizing via python-unicodedata library.
        
        ---------
        :param str text: Text for normalization
        :param bool remove_space: Whether remove empty spaces or not (defaults to True)
        :param bool do_lower_case: Whether do lower case or not (defaults to True)
        :param str normalize_function: Unicodedata normalize function. Either "NFC", "NFKC", "NFD" or "NFKD". (defaults to "NFKC")
        :param bool is_turkish: Whether text is in Turkish or not (defaults to True)
        :param bool use_zemberek: Whether to use Zemberek-Python's normalizer. Always do lowercase inside (defaults to True)
        :return: Normalized text
        """
        outputs: str = text

        if remove_space:
            outputs = " ".join(outputs.strip().split())

        outputs = unicodedata.normalize(normalize_function, outputs)
        outputs = "".join([c for c in outputs if not unicodedata.combining(c)])

        if use_zemberek:
            sentences = self.zemberek_extractor.from_paragraph(outputs)
            normalized_sentences = []
            for sentence in sentences:
                normalized_sentences.append(self.zemberek_normalizer.normalize(sentence))
            outputs = "".join(normalized_sentences)

        if do_lower_case:
            if is_turkish:
                outputs = outputs.replace('\u0049', '\u0131')  # I -> ı
                outputs = outputs.replace('\u0130', '\u0069')  # İ -> i

            outputs = outputs.casefold()

        return outputs


def special_token(x):
    """
    Add special token if text contains upper chars.
    
    ---------
    param x: Text
    return: Adjusted text
    """
    for term in second_hand_terms:
        if term in x:
            x = x.replace(term, f"^s^^ {term}")
    for term in risky_terms:
        if term in x:
            x = x.replace(term, f"#r## {term}")
    return x


# Word Count Feature
def feature_wordcount(x):
    """
    Count the word in a text using string split() function. If the length condition met, add special token
    
    ---------
    param x: Text
    return: Adjusted text
    """
    length = len(x.split())
    if length > 1000:
        return "+++ " + x
    elif length > 1500:
        return "+++++ " + x
    elif length > 2000:
        return "++++++" + x
    return x


def preprocess_text(textcol,
                    prevent_bias=0):
    """
    Remove punctuations, prevent the bias by bias level
    
    ---------
    param textcol: Text
    param prevent_bias: bias level. 2 means fully-unbiased, 1 means casing-unbiased, 0 means none bias prevention mechanism is being executed
    return: Adjusted text
    """
    # textcol.values[:] = [" ".join(elm.strip().split()) for elm in tqdm(textcol.values)]
    # textcol.values[:] = [unicodedata.normalize("NFKC", elm) for elm in tqdm(textcol.values)]
    # textcol.values[:] = ["".join([c for c in elm if not unicodedata.combining(c)]) for elm in tqdm(textcol.values)]

    # Removing punctuations
    textcol = textcol.apply(lambda x: x.translate(remove_punc))

    # Casing-Unbiased and Fully-Unbiased Flow
    if prevent_bias > 0:
        textcol = textcol.str.lower()

    textcol = textcol.apply(special_token)
    textcol = textcol.apply(feature_wordcount)

    return textcol






