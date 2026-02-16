FROM node:20.12.2

WORKDIR /app

RUN npm install --global corepack@latest
RUN corepack enable && corepack prepare pnpm@latest --activate

COPY package.json ./
COPY pnpm-lock.yaml ./

RUN pnpm install
