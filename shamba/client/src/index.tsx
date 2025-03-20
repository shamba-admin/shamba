import { configureStore } from "@reduxjs/toolkit";
import { render } from "preact";
import { Suspense } from "preact/compat";
import queryString from "query-string";
import { Provider } from "react-redux";
import { connectRoutes } from "redux-first-router";
import { createEpicMiddleware } from "redux-observable";
import { persistReducer, persistStore } from "redux-persist";
import { PersistGate } from "redux-persist/integration/react";
import storage from "redux-persist/lib/storage";
import createHistory from "rudy-history/createHashHistory";

import "~/_common/i18n/translations";
import App from "~/main/components/App";
import rootEpic from "~/main/epics/epics";
import { createPhoenixSocketMiddleware } from "~/main/middleware/phoenix";
import mainReducer from "~/main/reducers/mainReducer";
import pageReducer from "~/main/reducers/pageReducer";
import { routesMap } from "~/main/routesMap";

const firstRouter = connectRoutes(routesMap, {
  querySerializer: queryString,
  createHistory: createHistory.default,
  initialDispatch: false,
});

const epicMiddleware = createEpicMiddleware();
const phoenixSocketMiddleware = createPhoenixSocketMiddleware("/socket");

const statePersistConfig = {
  key: "state",
  storage: storage.default,
  whitelist: ["user", "token"],
};

const store = configureStore({
  reducer: {
    state: persistReducer(statePersistConfig, mainReducer),
    location: firstRouter.reducer,
    page: pageReducer,
  },
  middleware: () => [
    epicMiddleware,
    firstRouter.middleware,
    phoenixSocketMiddleware,
  ],
  enhancers: (getDefaultEnhancers) =>
    getDefaultEnhancers().concat(firstRouter.enhancer),
});

epicMiddleware.run(rootEpic);

export const persistor = persistStore(store, {}, () => {
  // Ensures ALL route actions are run after redux-observables
  // subscribes to the store
  firstRouter.initialDispatch();
});

render(
  <Provider store={store}>
    <PersistGate loading={null} persistor={persistor}>
      <Suspense>
        <App />
      </Suspense>
    </PersistGate>
  </Provider>,
  document.getElementById("root"),
);
