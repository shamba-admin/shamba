import i18n from "i18next";
import { initReactI18next } from "react-i18next";

export const I18N = "i18n";

export const translations = {
  fr: {
    translation: {},
  },
};

export const i18nPromise = i18n
  .use(initReactI18next) // passes i18n down to react-i18next
  .init({
    resources: translations,
    fallbackLng: "en",
  });

export default i18nPromise;
