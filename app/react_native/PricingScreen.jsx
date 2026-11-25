import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  SafeAreaView,
} from 'react-native';
import { fetchPriceRecommendation } from './apiService';

export default function PricingScreen() {
  const [listingId, setListingId] = useState(101);
  const [mode, setMode] = useState('pricing_with_insight');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);

  const loadRecommendation = async () => {
    setLoading(true);
    try {
      const res = await fetchPriceRecommendation(listingId, mode);
      setData(res);
    } catch (err) {
      console.warn('API fetch failed, loading default view:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadRecommendation();
  }, [listingId, mode]);

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Airbnb Dynamic Pricing</Text>
          <Text style={styles.headerSubtitle}>Pricing Engine & LLM Intelligence (Member 2)</Text>
        </View>

        {/* Listing Selection Pills */}
        <View style={styles.selectionSection}>
          <Text style={styles.sectionLabel}>Select Demonstration Listing:</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.pillsRow}>
            {[
              { id: 101, label: '101: Williamsburg 1BR (SVD)' },
              { id: 102, label: '102: East Village Studio' },
              { id: 103, label: '103: Park Slope 2BR Loft' },
              { id: 105, label: '105: Cold-Start Fallback' },
            ].map((item) => (
              <TouchableOpacity
                key={item.id}
                onPress={() => setListingId(item.id)}
                style={[styles.pill, listingId === item.id && styles.pillActive]}
              >
                <Text style={[styles.pillText, listingId === item.id && styles.pillTextActive]}>
                  {item.label}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        {/* Execution Mode Selector */}
        <View style={styles.modeContainer}>
          {['pricing', 'pricing_with_insight', 'insight'].map((m) => (
            <TouchableOpacity
              key={m}
              onPress={() => setMode(m)}
              style={[styles.modeTab, mode === m && styles.modeTabActive]}
            >
              <Text style={[styles.modeTabText, mode === m && styles.modeTabTextActive]}>
                {m === 'pricing' ? 'Price Only' : m === 'insight' ? 'LLM Only' : 'Price + LLM'}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {loading && (
          <View style={styles.loadingBox}>
            <ActivityIndicator size="large" color="#FF385C" />
            <Text style={styles.loadingText}>Running Market Aggregation & LLM...</Text>
          </View>
        )}

        {/* Target Listing Details */}
        {data && data.target_listing && (
          <View style={styles.targetCard}>
            <Text style={styles.targetName}>{data.target_listing.name}</Text>
            <Text style={styles.targetMeta}>
              {data.target_listing.neighborhood} • {data.target_listing.room_type}
            </Text>
          </View>
        )}

        {/* Hero Price Card (if mode includes pricing) */}
        {data && mode !== 'insight' && (
          <View style={styles.heroCard}>
            <View style={styles.heroHeader}>
              <Text style={styles.heroLabel}>RECOMMENDED NIGHTLY RATE</Text>
              <View style={[styles.badge, data.fallback_used ? styles.badgeFallback : styles.badgeSvd]}>
                <Text style={styles.badgeText}>
                  {data.fallback_used ? 'Content Fallback' : 'SVD Match'}
                </Text>
              </View>
            </View>

            <Text style={styles.priceHero}>
              ${Math.round(data.recommended_price || 0)}
              <Text style={styles.pricePerNight}> / night</Text>
            </Text>

            <View style={styles.rangeContainer}>
              <Text style={styles.rangeText}>Min: ${Math.round(data.price_min || 0)}</Text>
              <Text style={styles.rangeTextMedian}>Median: ${Math.round(data.price_median || 0)}</Text>
              <Text style={styles.rangeText}>Max: ${Math.round(data.price_max || 0)}</Text>
            </View>
            <Text style={styles.comparablesCount}>
              Aggregated from {data.comparable_count} comparable peer listings
            </Text>
          </View>
        )}

        {/* LLM Insight Card */}
        {data && data.insight && mode !== 'pricing' && (
          <View style={styles.insightCard}>
            <View style={styles.insightHeader}>
              <Text style={styles.insightTitle}>🤖 AI Revenue Advisor (LLM)</Text>
              <Text style={styles.insightTierBadge}>{data.insight.market_tier}</Text>
            </View>
            <Text style={styles.insightHeadline}>{data.insight.headline}</Text>
            <Text style={styles.subHeading}>PRICING RATIONALE</Text>
            <Text style={styles.bodyText}>{data.insight.pricing_rationale}</Text>
            
            <Text style={styles.subHeading}>HOST OPTIMIZATION TIPS</Text>
            {data.insight.actionable_tips.map((tip, idx) => (
              <View key={idx} style={styles.tipRow}>
                <Text style={styles.tipBullet}>✓</Text>
                <Text style={styles.tipText}>{tip}</Text>
              </View>
            ))}

            <View style={styles.confidenceFooter}>
              <Text style={styles.confidenceText}>{data.insight.confidence_commentary}</Text>
            </View>
          </View>
        )}

        {/* Comparable Listings Preview */}
        {data && data.comparables && mode !== 'insight' && (
          <View style={styles.compsSection}>
            <Text style={styles.sectionTitle}>Top Comparable Listings</Text>
            {data.comparables.map((c) => (
              <View key={c.id} style={styles.compRow}>
                <View style={styles.compInfo}>
                  <Text style={styles.compName} numberOfLines={1}>{c.name}</Text>
                  <Text style={styles.compMeta}>{c.neighborhood} • Sim: {Math.round(c.similarity_score * 100)}%</Text>
                </View>
                <Text style={styles.compPrice}>${Math.round(c.price)}</Text>
              </View>
            ))}
          </View>
        )}

      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F8FAFC' },
  scrollContent: { padding: 16, paddingBottom: 40 },
  header: { marginBottom: 16 },
  headerTitle: { fontSize: 22, fontWeight: '800', color: '#0F172A' },
  headerSubtitle: { fontSize: 13, color: '#64748B', fontWeight: '500' },
  selectionSection: { marginBottom: 16 },
  sectionLabel: { fontSize: 12, fontWeight: '700', color: '#475569', marginBottom: 8 },
  pillsRow: { flexDirection: 'row' },
  pill: { paddingHorizontal: 14, paddingVertical: 8, backgroundColor: '#E2E8F0', borderRadius: 20, marginRight: 8 },
  pillActive: { backgroundColor: '#FF385C' },
  pillText: { fontSize: 12, fontWeight: '600', color: '#334155' },
  pillTextActive: { color: '#FFFFFF' },
  modeContainer: { flexDirection: 'row', backgroundColor: '#E2E8F0', borderRadius: 12, padding: 4, marginBottom: 16 },
  modeTab: { flex: 1, paddingVertical: 8, alignItems: 'center', borderRadius: 8 },
  modeTabActive: { backgroundColor: '#FFFFFF' },
  modeTabText: { fontSize: 12, fontWeight: '600', color: '#64748B' },
  modeTabTextActive: { color: '#FF385C' },
  loadingBox: { padding: 20, alignItems: 'center' },
  loadingText: { fontSize: 13, color: '#64748B', marginTop: 8 },
  targetCard: { backgroundColor: '#FFFFFF', padding: 14, borderRadius: 16, marginBottom: 14, borderWidth: 1, borderColor: '#E2E8F0' },
  targetName: { fontSize: 15, fontWeight: '700', color: '#0F172A' },
  targetMeta: { fontSize: 12, color: '#64748B', marginTop: 4 },
  heroCard: { backgroundColor: '#0F172A', padding: 20, borderRadius: 20, marginBottom: 16 },
  heroHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  heroLabel: { fontSize: 11, fontWeight: '800', color: '#FB7185' },
  badge: { paddingHorizontal: 8, paddingVertical: 4, borderRadius: 12 },
  badgeSvd: { backgroundColor: 'rgba(16, 185, 129, 0.2)' },
  badgeFallback: { backgroundColor: 'rgba(245, 158, 11, 0.2)' },
  badgeText: { fontSize: 10, fontWeight: '700', color: '#34D399' },
  priceHero: { fontSize: 44, fontWeight: '900', color: '#FFFFFF', marginVertical: 8 },
  pricePerNight: { fontSize: 16, fontWeight: '500', color: '#94A3B8' },
  rangeContainer: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 12, paddingTop: 12, borderTopWidth: 1, borderTopColor: '#334155' },
  rangeText: { fontSize: 12, color: '#94A3B8' },
  rangeTextMedian: { fontSize: 12, fontWeight: '700', color: '#FB7185' },
  comparablesCount: { fontSize: 11, color: '#64748B', marginTop: 8 },
  insightCard: { backgroundColor: '#FFFFFF', padding: 18, borderRadius: 18, marginBottom: 16, borderWidth: 1, borderColor: '#E2E8F0' },
  insightHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  insightTitle: { fontSize: 14, fontWeight: '700', color: '#1E1B4B' },
  insightTierBadge: { fontSize: 11, fontWeight: '700', color: '#4F46E5', backgroundColor: '#EEF2FF', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 10 },
  insightHeadline: { fontSize: 14, fontWeight: '600', color: '#1E293B', backgroundColor: '#F8FAFC', padding: 10, borderRadius: 10, marginBottom: 12 },
  subHeading: { fontSize: 11, fontWeight: '800', color: '#94A3B8', marginTop: 8, marginBottom: 4 },
  bodyText: { fontSize: 12, color: '#475569', lineHeight: 18 },
  tipRow: { flexDirection: 'row', alignItems: 'flex-start', marginTop: 6 },
  tipBullet: { color: '#FF385C', fontWeight: 'bold', marginRight: 6 },
  tipText: { fontSize: 12, color: '#334155', flex: 1, lineHeight: 16 },
  confidenceFooter: { marginTop: 12, paddingTop: 10, borderTopWidth: 1, borderTopColor: '#F1F5F9' },
  confidenceText: { fontSize: 11, color: '#64748B', fontStyle: 'italic' },
  compsSection: { backgroundColor: '#FFFFFF', padding: 16, borderRadius: 16, borderWidth: 1, borderColor: '#E2E8F0' },
  sectionTitle: { fontSize: 14, fontWeight: '700', color: '#0F172A', marginBottom: 12 },
  compRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: '#F1F5F9' },
  compInfo: { flex: 1, marginRight: 10 },
  compName: { fontSize: 13, fontWeight: '600', color: '#1E293B' },
  compMeta: { fontSize: 11, color: '#94A3B8', marginTop: 2 },
  compPrice: { fontSize: 14, fontWeight: '800', color: '#0F172A' },
});
