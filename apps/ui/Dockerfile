# Local/OrbStack use. NEXT_PUBLIC_API_BASE_URL is inlined into the client
# bundle at build time (Next.js convention for NEXT_PUBLIC_* vars) -- it
# must be the browser-reachable API origin (e.g. http://localhost:8000),
# not a docker-network-internal hostname, since fetches run in the user's
# browser, not inside this container.
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
RUN npm run build

FROM node:20-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/next.config.ts ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

EXPOSE 3000
CMD ["npm", "start"]
